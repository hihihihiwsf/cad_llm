"""
ByT5 pytorch lightning model
"""

try:
    import comet_ml  # Import before torch
except ImportError:
    pass
import torch
import torch.optim as optim
# import lightning.pytorch as pl
import pytorch_lightning as pl
from transformers import T5Config, AutoTokenizer
from transformers.modeling_utils import unwrap_model
import sys
from models.vis_recon import VisRecon
sys.path.insert(0, '/home/ec2-user/SageMaker/efs/code/cad_llm')
from metrics import calculate_accuracy, calculate_first_ent_accuracy, calculate_validity, calculate_f1
from util import get_quantized_range
from geometry.parse import get_curves, get_point_entities
from geometry.visualization import visualize_batch
from pathlib import Path
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from PIL import Image
# import clip
import numpy as np
from transformers import CLIPVisionModelWithProjection, CLIPVisionModel, AutoImageProcessor
from models.modeling_vit_mae import ViTMAEForPreTraining
from models.modeling_vlt5 import T5ForConditionalGeneration
from geometry.visualize_vit import Visualize_VIT

import torch.nn as nn
from transformers.optimization import Adafactor, AdafactorSchedule, get_cosine_schedule_with_warmup
from copy import deepcopy

from models.modeling_vit_mae import get_2d_sincos_pos_embed, ViTMAEDecoderOutput

from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)

from torchvision.ops.focal_loss import sigmoid_focal_loss
from lion_pytorch import Lion

class BlipTextPooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
    
class ImageTransform(nn.Module):
    def __init__(self, vis_model_config, text_model_config):
        super().__init__()
        self.patch_num = int(vis_model_config.image_size/vis_model_config.patch_size)

        self.downsample = torch.nn.Linear(self.patch_num*self.patch_num +1, self.patch_num)
        self.gelu = torch.nn.GELU()
        
        self.mapper = torch.nn.Linear(vis_model_config.hidden_size, text_model_config.d_model)
        self.activation = nn.Tanh()
        self.layernorm = torch.nn.LayerNorm(text_model_config.d_model, eps=1e-5)
        
    def forward(self, img_hidden_state):
        _image_embeds = img_hidden_state.permute(0,2,1)
        '''patch embedding downsample 196 to 14'''
        image_for_llm = self.activation(self.downsample(_image_embeds).permute(0,2,1))
        image_for_llm = self.mapper(image_for_llm)
        image_for_llm = self.activation(self.layernorm(image_for_llm))
        
        return image_for_llm
        

class ByT5Model(pl.LightningModule):
    def __init__(self, args, vit_mae,num_train_steps):
        super().__init__()
        self.save_hyperparameters()
        
        self.num_train_steps = num_train_steps
        self.args = args
        self.lr = self.args.lr
        self.batch_size = self.args.batch_size  # to fix logging warning
        self.quantization_bits = 6  # Hard code for now
        self.quantized_range = get_quantized_range(self.quantization_bits)
        self.box_lim = max(self.quantized_range)  # for visualization
        
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        # self.tokenizer.add_special_tokens(["<IMAGE>"])
        
        model = T5ForConditionalGeneration.from_pretrained(args.model_name)
        self.model = model
        
        self.projection_dim = 1024
        self.text_embed_dim = self.model.config.d_model
        
        self.textpooler = BlipTextPooler(self.text_embed_dim )
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        

        self.vit_mae = vit_mae
        if vit_mae is not None:
            self.vit_mae = vit_mae
        else:
            m = VisRecon(args=args)
            # m.load_from_checkpoint('s3://cad-llm-katzm/jobs/vitmae_deepmind/checkpoints/best.ckpt')
            m.load_from_checkpoint('s3://cad-llm-katzm/checkpoints/vitmae_sg/best.ckpt')
            self.vit_mae = m.model 
            del m
        
        self.vis_model = self.vit_mae
        self.vis_model.config.mask_ratio = 0.
        #self.vis_model.requires_grad_(False)
        self.vitmae_preprocess = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")

        self.img_transform = ImageTransform(self.vis_model.config, self.model.config)
        
        self.vision_embed_dim = self.vis_model.config.hidden_size
        self.vision_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
       
        self.back_mapper = torch.nn.Linear(self.model.config.d_model, self.vis_model.config.hidden_size)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592) #logit_scale_init_value=2.6592
        self.activation = nn.Tanh()
        #self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        
        #opt1, opt2 = self.optimizers()
        cols = ["attention_mask", "labels"]
        model_batch = {col: val for col, val in batch.items() if col in cols}

        '''img encoder'''
        # image_features = self.clip_model.encode_image(batch['images'])
        # batch['images'] = self.vitmae_preprocess(batch['images'], return_tensors="pt")
        oi = self.vis_model.vit(batch['images'], output_hidden_states=True) 
        img_last_hidden_state = oi['last_hidden_state']
        
        image_for_llm = self.img_transform(img_last_hidden_state)
        image_embeds = self.vision_projection(img_last_hidden_state[:,0]) #for contrastive learning
        

        '''txt encoder'''
        txt_encoder_output = self.model.encoder(batch['input_ids'], batch['attention_mask'],output_hidden_states=True)
        _txt_embeds = txt_encoder_output[0]
        
        text_embeds = self.textpooler(_txt_embeds)
        text_embeds = self.text_projection(text_embeds)
        
        '''fuse encoder output'''
        output_embed = torch.concatenate((image_for_llm, _txt_embeds), dim=1)
        # input_embed = torch.concatenate((imm, image_for_llm.unsqueeze(1), code, txt_embeddings), dim=1)
        model_batch['encoder_outputs_embeds'] = output_embed


        # adding ones to attention_mask
        att = model_batch['attention_mask']
        model_batch['attention_mask'] = torch.cat((torch.ones(att.shape[0], image_for_llm.shape[1]).to(self.device), att), dim=1)
        # model_batch['attention_mask'] = torch.cat((torch.ones(att.shape[0], code.shape[1]+imm.shape[1]+1).to(self.device), att), dim=1)
        
        decoder_input_ids = self.model._shift_right(batch['labels'])
        # Decode
        decoder_outputs = self.model.decoder(
            input_ids =decoder_input_ids,
            encoder_hidden_states=model_batch['encoder_outputs_embeds'],
            encoder_attention_mask=model_batch['attention_mask'],
            output_hidden_states=True)

        sequence_output = decoder_outputs[0]

        '''text reconstruction loss'''
        # Set device for model parallelism
        if self.model.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.model.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.model.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.model.model_dim**-0.5)

        lm_logits = self.model.lm_head(sequence_output)

        loss = None
        if model_batch['labels'] is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            txt_loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), model_batch['labels'].view(-1))
        
        
        '''image decoder pixel loss'''
        img_hidden_state = self.img_transform.layernorm(output_embed)
        self.post_layernorm=self.vis_model.vit.layernorm
        img_hidden_state = self.activation(self.post_layernorm(self.back_mapper(img_hidden_state)))
        self.mask = nn.Parameter(torch.zeros(img_hidden_state.shape[0], img_last_hidden_state.shape[1]-img_hidden_state.shape[1], img_hidden_state.shape[2])).to(img_hidden_state.device)
        img_hidden_state = torch.concat((img_hidden_state, self.mask), dim=1)
        #img_hidden_state = self.back_patch(img_hidden_state.permute(0,2,1))
        
        img_res = self.vis_model.decoder(img_hidden_state, ids_restore=oi.ids_restore)
        img_loss = self.forward_focal_loss(batch['output_images'], img_res.logits) #img_res.logits: #(bs, 196, v_dim)
        
        '''image text contrastive loss'''
        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        similarity = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        #logits_per_image = logits_per_text.t() # = similarity.t()
        
        contrastive_loss = nn.functional.cross_entropy(similarity, torch.arange(len(similarity), device=similarity.device))
        
        loss = (txt_loss + img_loss + contrastive_loss) / 3.0
        self.log("img_loss", img_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=self.batch_size, sync_dist=True)
        self.log("contrastive_loss", contrastive_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=self.batch_size, sync_dist=True)
        self.log("txt_loss", txt_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=self.batch_size, sync_dist=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=self.batch_size, sync_dist=True)
        
        # if batch_idx % 15==0:
        #     sch1.step()
        #     sch2.step()
        # opt1.zero_grad()
        # opt2.zero_grad()
        # self.manual_backward(loss)
        # opt1.step()
        # opt2.step()

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.evaluation_process(batch, batch_idx, validate='val')
        return loss
    
    def test_step(self,batch, batch_idx):
        loss = self.evaluation_process(batch, batch_idx, validate='test')
        return loss
        
        
    def evaluation_process(self, batch, batch_idx, validate):
        cols = ["attention_mask", "labels"]
        model_batch = {col: val for col, val in batch.items() if col in cols}

        '''img encoder'''
        # image_features = self.clip_model.encode_image(batch['images'])
        # batch['images'] = self.vitmae_preprocess(batch['images'], return_tensors="pt")
        oi = self.vis_model.vit(batch['images'], output_hidden_states=True) 
        img_last_hidden_state = oi['last_hidden_state']
        
        image_for_llm = self.img_transform(img_last_hidden_state)
        image_embeds = self.vision_projection(img_last_hidden_state[:,0]) #for contrastive learning
        

        '''txt encoder'''
        txt_encoder_output = self.model.encoder(batch['input_ids'], batch['attention_mask'],output_hidden_states=True)
        _txt_embeds = txt_encoder_output[0]
        
        text_embeds = self.textpooler(_txt_embeds)
        text_embeds = self.text_projection(text_embeds)
    
        output_embed = torch.concatenate((image_for_llm, _txt_embeds), dim=1)
        # input_embed = torch.concatenate((imm, image_for_llm.unsqueeze(1), code, txt_embeddings), dim=1)
        model_batch['encoder_outputs_embeds'] = output_embed


        # adding ones to attention_mask
        att = model_batch['attention_mask']
        model_batch['attention_mask'] = torch.cat((torch.ones(att.shape[0], image_for_llm.shape[1]).to(self.device), att), dim=1)
        # model_batch['attention_mask'] = torch.cat((torch.ones(att.shape[0], code.shape[1]+imm.shape[1]+1).to(self.device), att), dim=1)
        
        decoder_input_ids = self.model._shift_right(batch['labels'])
        # Decode
        decoder_outputs = self.model.decoder(
            input_ids =decoder_input_ids,
            encoder_hidden_states=model_batch['encoder_outputs_embeds'],
            encoder_attention_mask=model_batch['attention_mask'],
            output_hidden_states=True)

        sequence_output = decoder_outputs[0]

        '''text reconstruction loss'''
        # Set device for model parallelism
        if self.model.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.model.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.model.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.model.model_dim**-0.5)

        lm_logits = self.model.lm_head(sequence_output)

        loss = None
        if model_batch['labels'] is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            txt_loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), model_batch['labels'].view(-1))
        
        
        '''image decoder pixel loss'''
        img_hidden_state = self.img_transform.layernorm(output_embed)
        self.post_layernorm=self.vis_model.vit.layernorm
        img_hidden_state = self.activation(self.post_layernorm(self.back_mapper(img_hidden_state)))
        self.mask = nn.Parameter(torch.zeros(img_hidden_state.shape[0], img_last_hidden_state.shape[1]-img_hidden_state.shape[1], img_hidden_state.shape[2])).to(img_hidden_state.device)
        img_hidden_state = torch.concat((img_hidden_state, self.mask), dim=1)
        #img_hidden_state = self.back_patch(img_hidden_state.permute(0,2,1))
        
        img_res = self.vis_model.decoder(img_hidden_state, ids_restore=oi.ids_restore)
        img_loss = self.forward_loss(batch['output_images'], img_res.logits) #img_res.logits: #(bs, 196, v_dim)
        
        '''image text contrastive loss'''
        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        similarity = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        #logits_per_image = logits_per_text.t() # = similarity.t()
        
        contrastive_loss = nn.functional.cross_entropy(similarity, torch.arange(len(similarity), device=similarity.device))
        
        loss = (txt_loss + img_loss + contrastive_loss) / 3.0
        self.log(f"{validate}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=self.batch_size, sync_dist=True)
        
        # Generate and process samples
        batch['attention_mask'] = model_batch['attention_mask']
        encoder_outputs = BaseModelOutput(last_hidden_state=output_embed)
        batch['encoder_outputs'] = encoder_outputs
        
        self.generate_samples(batch)

        # Calculate metrics
        top1_full_sketch = calculate_accuracy(samples=batch["point_samples"], labels=batch["point_labels"])

        self.log(f"{validate}_top1_full_sketch", top1_full_sketch, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=self.batch_size, sync_dist=True)

        top1_ent = calculate_first_ent_accuracy(samples=batch["point_samples"], labels=batch["point_labels"])

        self.log(f"{validate}_top1_ent", top1_ent, on_step=False, on_epoch=True, prog_bar=True, logger=True,
            batch_size=self.batch_size, sync_dist=True)
        # # Convert string entities to curves and check validity
        validity = calculate_validity(batch_sample_curves=batch["sample_curves"])
        self.log(f"{validate}_validity", validity, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=self.batch_size, sync_dist=True)

        f1 = calculate_f1(samples=batch["point_samples"], labels=batch["point_labels"])
        self.log(f"{validate}_f1", f1, on_step=False, on_epoch=True, prog_bar=True, logger=True,
            batch_size=self.batch_size, sync_dist=True)

        return loss

    def generate_samples(self, batch):
        # Recursively unwrap the model from potential distributed training containers
        generate_func = unwrap_model(self.model).generate
        batch["samples"] = generate_func(encoder_outputs=batch["encoder_outputs"], attention_mask=batch["attention_mask"],
                                         do_sample=False, max_new_tokens=self.args.max_length+10)

        batch["string_samples"] = self.tokenizer.batch_decode(batch["samples"], skip_special_tokens=True)
        batch["string_labels"] = [sketch["output_text"].replace ('</s>', '') for sketch in batch["sketches"]]

        batch["point_samples"] = [get_point_entities(string_sample) for string_sample in batch["string_samples"]]
        batch["point_labels"] = [get_point_entities(string_label) for string_label in batch["string_labels"]]

        batch["sample_curves"] = [get_curves(point_sample) for point_sample in batch["point_samples"]]

    def forward_loss(self, pixel_values, pred):
        """
        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                Pixel values.
            pred (`torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
                Predicted pixel values.
            mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                Tensor indicating which patches are masked (1) and which are not (0).

        Returns:
            `torch.FloatTensor`: Pixel reconstruction loss.
        """
        target = self.vis_model.patchify(pixel_values)
        if self.vis_model.config.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target)**2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        #loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        loss = loss.mean()
        return loss

    def forward_focal_loss(self, pixel_values, pred):
        gamma = 2
        target = self.vis_model.patchify(pixel_values)
        if self.vis_model.config.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5
            
        # abs_error = torch.abs(pred - target)
        # normalized_error = abs_error / 64  # or use torch.sigmoid(abs_error)
        # focal_weight = (1 - normalized_error) ** gamma
        # focal_loss = focal_weight * abs_error
        focal_loss = sigmoid_focal_loss(pred, target, alpha=0.25,gamma=2,reduction='mean')
        return focal_loss
    
    def log_samples(self, batch, batch_idx):
        label_curves = [get_curves(point_label) for point_label in batch["point_labels"]]

        point_inputs = [get_point_entities(sketch["input_text"]) for sketch in batch["sketches"]]
        input_curves = [get_curves(point_input) for point_input in point_inputs]

        fig = visualize_batch(input_curves=input_curves, label_curves=label_curves,
                              sample_curves=batch["sample_curves"], box_lim=self.box_lim + 3)
        fig_path = Path(self.args.samples_dir) / f"epoch_{self.current_epoch}_batch_{batch_idx}.png"
        fig.savefig(fig_path)

    def configure_optimizers(self):
        
        #optimizer = Adafactor(params1+params2, scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
        #opt = Lion(params1+params2, lr=self.lr, weight_decay=1e-2)
        
        optimizer = optim.AdamW(self.trainer.model.parameters(), lr=self.lr)
        if not self.args.cosinedecay and not self.args.adafactor:
            return optimizer
        
        if self.args.cosinedecay:    
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_train_steps, eta_min=self.lr * 0.1)
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=4,sefsdfsdf verbose=True)
        
        if self.args.adafactor:
            optimizer = Adafactor(self.trainer.model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
            scheduler = AdafactorSchedule(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        }
            
        #scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=int(self.args.epochs * 1.15), verbose=True)
        #scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=int(self.args.epochs * 1.15), verbose=True)

        # scheduler_A = {
        #     "optimizer": optimizer1,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "interval": "epoch",
        #         "frequency": 1,
        #     }
        # }
        # scheduler_B = {
        #     "optimizer": optimizer2,
        #     "lr_scheduler": {
        #         "scheduler": scheduler2,
        #         "interval": "epoch",
        #         "frequency": 1,
        #     }
        # }
    
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=4,verbose=True)
        #lr_scheduler = AdafactorSchedule(optimizer1)


