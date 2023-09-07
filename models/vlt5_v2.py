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
from transformers.optimization import Adafactor, AdafactorSchedule
from copy import deepcopy

from models.modeling_vit_mae import get_2d_sincos_pos_embed, ViTMAEDecoderOutput

class ImageDecoderModel(pl.LightningModule):
    def __init__(self, in_hidden_size, ViTMAELayer, config, num_patches):
        super().__init__()
        self.decoder_embed = nn.Linear(in_hidden_size, config.decoder_hidden_size, bias=True)
        #self.mask_token = nn.Parameter(torch.zeros(1, 1, config.decoder_hidden_size))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.decoder_hidden_size), requires_grad=False
        )  # fixed sin-cos embedding

        decoder_config = deepcopy(config)
        decoder_config.hidden_size = config.decoder_hidden_size
        decoder_config.num_hidden_layers = config.decoder_num_hidden_layers
        decoder_config.num_attention_heads = config.decoder_num_attention_heads
        decoder_config.intermediate_size = config.decoder_intermediate_size
        self.decoder_layers = nn.ModuleList(
            [ViTMAELayer(decoder_config) for _ in range(config.decoder_num_hidden_layers)]
        )

        self.decoder_norm = nn.LayerNorm(config.decoder_hidden_size, eps=config.layer_norm_eps)
        self.decoder_pred = nn.Linear(
            config.decoder_hidden_size, config.patch_size**2 * config.num_channels, bias=True
        )  # encoder to decoder
        self.gradient_checkpointing = False
        self.config = config
        self.initialize_weights(num_patches)

    def initialize_weights(self, num_patches):
        # initialize (and freeze) position embeddings by sin-cos embedding
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], int(num_patches**0.5), add_cls_token=True
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=self.config.initializer_range)

    def forward(
        self,
        hidden_states,
        ids_restore,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        # embed tokens
        x = self.decoder_embed(hidden_states)

        # append mask tokens to sequence
        # mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        # x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        # x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        # x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        hidden_states = x + self.decoder_pos_embed

        # apply Transformer layers (blocks)
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.decoder_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    None,
                )
            else:
                layer_outputs = layer_module(hidden_states, head_mask=None, output_attentions=output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.decoder_norm(hidden_states)

        # predictor projection
        logits = self.decoder_pred(hidden_states)

        # remove cls token
        logits = logits[:, 1:, :]

        if not return_dict:
            return tuple(v for v in [logits, all_hidden_states, all_self_attentions] if v is not None)
        return ViTMAEDecoderOutput(
            logits=logits,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class ByT5Model(pl.LightningModule):
    def __init__(self, args, vit_mae):
        super().__init__()
        self.save_hyperparameters()


        model = T5ForConditionalGeneration.from_pretrained(args.model_name)

        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        self.vitmae_preprocess = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
        # self.tokenizer.add_special_tokens(["<IMAGE>"])

        self.args = args

        self.lr = self.args.lr
        self.batch_size = self.args.batch_size  # to fix logging warning
        self.quantization_bits = 6  # Hard code for now
        self.quantized_range = get_quantized_range(self.quantization_bits)
        self.box_lim = max(self.quantized_range)  # for visualization
        

        self.vit_mae = vit_mae
        if vit_mae is not None:
            self.vit_mae = vit_mae
        else:
            m = VisRecon(args=args)
            #m.load_from_checkpoint('s3://cad-llm-katzm/jobs/vitmae_deepmind/checkpoints/best.ckpt')
            m.load_from_checkpoint('s3://cad-llm-katzm/checkpoints/vitmae_sg/best.ckpt')
            self.vit_mae = m.model 
            del m
        
        self.vis_model = self.vit_mae
        self.vis_model.config.mask_ratio = 0.
        #self.vis_model.requires_grad_(False)

        self.mapper = torch.nn.Linear(self.vis_model.config.hidden_size, self.model.config.d_model)
        self.back_mapper = torch.nn.Linear(self.model.config.d_model, self.vis_model.config.hidden_size)

        self.post_layernorm = self.vis_model.vit.layernorm
        self.layernorm = torch.nn.LayerNorm(self.model.config.d_model, eps=1e-12)

        self.patch_num = int(self.vis_model.config.image_size/self.vis_model.config.patch_size)

        self.embed_patch = torch.nn.Linear(self.patch_num*self.patch_num +1, self.patch_num)
        self.gelu = torch.nn.GELU()
        
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592) #logit_scale_init_value=2.6592


    def training_step(self, batch, batch_idx):
        cols = ["attention_mask", "labels"]
        model_batch = {col: val for col, val in batch.items() if col in cols}

        
        # image_features = self.clip_model.encode_image(batch['images'])
        # batch['images'] = self.vitmae_preprocess(batch['images'], return_tensors="pt")
        with torch.no_grad():
            oi = self.vis_model.vit(batch['images'], output_hidden_states=True) 

        '''owl vit'''
        last_hidden_state = oi['last_hidden_state']
        #pooled_output= last_hidden_state[:, 0, :]
        image_embeds = self.post_layernorm(last_hidden_state)
        
        '''patch embedding downsample 196 to 14'''
        image_embeds = image_embeds.permute(0,2,1)
        image_embeds = self.gelu(self.embed_patch(image_embeds).permute(0,2,1))

        image_for_llm = self.gelu(self.layernorm(self.mapper(image_embeds.float())))
        
        # image_for_llm = self.layernorm(image_for_llm)

        txt_embedder = self.model.get_input_embeddings()
        txt_embeddings = txt_embedder(batch['input_ids']) # size: (batch_size, seq_length, 1536)
        
        
        input_embed = torch.concatenate((image_for_llm, txt_embeddings), dim=1)
        # input_embed = torch.concatenate((imm, image_for_llm.unsqueeze(1), code, txt_embeddings), dim=1)
        model_batch['inputs_embeds'] = input_embed


        # adding ones to attention_mask
        att = model_batch['attention_mask']
        model_batch['attention_mask'] = torch.cat((torch.ones(att.shape[0], image_for_llm.shape[1]).to(self.device), att), dim=1)
        # model_batch['attention_mask'] = torch.cat((torch.ones(att.shape[0], code.shape[1]+imm.shape[1]+1).to(self.device), att), dim=1)

        batch['attention_mask'] = model_batch['attention_mask']
        batch['inputs_embeds'] = model_batch['inputs_embeds']

        outputs = self.model(**model_batch, output_hidden_states=True)
        decoder_hidden_state = outputs.decoder_hidden_states
        
        
        encoder_hidden_state = outputs.encoder_last_hidden_state
        img_hidden_state = self.layernorm(encoder_hidden_state)
        img_hidden_state = self.post_layernorm(self.back_mapper(img_hidden_state))
        mask = nn.Parameter(torch.zeros(img_hidden_state.shape[0], last_hidden_state.shape[1]-img_hidden_state.shape[1], img_hidden_state.shape[2])).to(img_hidden_state.device)
        img_hidden_state = torch.concat((img_hidden_state, mask), dim=1)
        #img_hidden_state = self.back_patch(img_hidden_state.permute(0,2,1))
        
        img_res = self.vis_model.decoder(img_hidden_state, ids_restore=oi.ids_restore)
        img_logits = img_res.logits
        img_loss = self.forward_loss(batch['output_images'], img_logits)
        
        loss = outputs.loss + img_loss  # CrossEntropyLoss(ignore_index=-100) between outputs.logits and labels
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True,
                 batch_size=self.batch_size, sync_dist=True)
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

        
        # image_features = self.clip_model.encode_image(batch['images'])
        # batch['images'] = self.vitmae_preprocess(batch['images'], return_tensors="pt")
        oi = self.vis_model.vit(batch['images'], output_hidden_states=True) 

        '''owl vit'''
        last_hidden_state = oi['last_hidden_state']
        #pooled_output= last_hidden_state[:, 0, :]
        image_embeds = last_hidden_state #self.post_layernorm(last_hidden_state)
        
        '''patch embedding downsample 196 to 14'''
        image_embeds = image_embeds.permute(0,2,1)
        image_embeds = self.gelu(self.embed_patch(image_embeds).permute(0,2,1))

        image_for_llm = self.gelu(self.layernorm(self.mapper(image_embeds.float())))
        # image_for_llm = self.layernorm(image_for_llm)

        txt_embedder = self.model.get_input_embeddings()
        txt_embeddings = txt_embedder(batch['input_ids']) # size: (batch_size, seq_length, 1536)
        
        
        input_embed = torch.concatenate((image_for_llm, txt_embeddings), dim=1)
        # input_embed = torch.concatenate((imm, image_for_llm.unsqueeze(1), code, txt_embeddings), dim=1)
        model_batch['inputs_embeds'] = input_embed


        # adding ones to attention_mask
        att = model_batch['attention_mask']
        model_batch['attention_mask'] = torch.cat((torch.ones(att.shape[0], image_for_llm.shape[1]).to(self.device), att), dim=1)
        # model_batch['attention_mask'] = torch.cat((torch.ones(att.shape[0], code.shape[1]+imm.shape[1]+1).to(self.device), att), dim=1)

        batch['attention_mask'] = model_batch['attention_mask']
        batch['inputs_embeds'] = model_batch['inputs_embeds']

        outputs = self.model(**model_batch, output_hidden_states=True)
        txt_logits = outputs.decoder_hidden_states[-1] #(bs, gen_len, t_dim)
        txt_loss = outputs.loss
        
        '''image pixel loss'''
        encoder_hidden_state = outputs.encoder_last_hidden_state
        
        img_hidden_state = self.layernorm(encoder_hidden_state)
        img_hidden_state = self.gelu(self.post_layernorm(self.back_mapper(img_hidden_state)))
        mask = nn.Parameter(torch.zeros(img_hidden_state.shape[0], last_hidden_state.shape[1]-img_hidden_state.shape[1], img_hidden_state.shape[2])).to(img_hidden_state.device)
        img_hidden_state = torch.concat((img_hidden_state, mask), dim=1)
        #img_hidden_state = self.back_patch(img_hidden_state.permute(0,2,1))
        
        img_res = self.vis_model.decoder(img_hidden_state, ids_restore=oi.ids_restore)
     
        img_loss = self.forward_loss(batch['output_images'], img_res.logits) #img_res.logits: #(bs, 196, v_dim)
    
        
        # '''contrastive loss'''
        # logit_scale = self.logit_scale.exp()
        # logits_per_text = torch.matmul(txt_logits, img_logits.t()) * logit_scale
        
        loss = txt_loss + img_loss 
        self.log(f"{validate}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=self.batch_size, sync_dist=True)
        
        
        # Generate and process samples
        self.generate_samples(batch)

        # Calculate metrics
        top1_full_sketch = calculate_accuracy(samples=batch["point_samples"], labels=batch["point_labels"])
        # mx = 0
        # for i,j in zip(batch['string_samples'], batch['string_labels']):
        #     out, l = i.split(";"), j.split(";")
        #     # label_all_ent = j.split(";")
        #     if set(out) == set(l):
        #         mx += 1
        # top1_full_sketch = mx/len(batch['string_labels'])
        self.log(f"{validate}_top1_full_sketch", top1_full_sketch, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=self.batch_size, sync_dist=True)

        top1_ent = calculate_first_ent_accuracy(samples=batch["point_samples"], labels=batch["point_labels"])

        # mx = 0
        # for i,j in zip(batch['string_samples'], batch['string_labels']):
        #     label_all_ent = j.split(";")
        #     first_ent = i.split(";")[0]
        #     if first_ent in label_all_ent:
        #         mx += 1
        # top1_ent = mx/len(batch['string_labels'])
        self.log(f"{validate}_top1_ent", top1_ent, on_step=False, on_epoch=True, prog_bar=True, logger=True,
            batch_size=self.batch_size, sync_dist=True)
        # # Convert string entities to curves and check validity
        validity = calculate_validity(batch_sample_curves=batch["sample_curves"])
        self.log(f"{validate}_validity", validity, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=self.batch_size, sync_dist=True)

        f1 = calculate_f1(samples=batch["point_samples"], labels=batch["point_labels"])
        self.log(f"{validate}_f1", f1, on_step=False, on_epoch=True, prog_bar=True, logger=True,
            batch_size=self.batch_size, sync_dist=True)

        # # # Plot sketches
        # if batch_idx < 5:
        #     self.log_samples(batch=batch, batch_idx=batch_idx)
        
        return loss


    
    def generate_samples(self, batch):
        # Recursively unwrap the model from potential distributed training containers
        generate_func = unwrap_model(self.model).generate
        batch["samples"] = generate_func(inputs_embeds=batch["inputs_embeds"], attention_mask=batch["attention_mask"],
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

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        #loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        loss = loss.mean()
        return loss
    
    def log_samples(self, batch, batch_idx):
        label_curves = [get_curves(point_label) for point_label in batch["point_labels"]]

        point_inputs = [get_point_entities(sketch["input_text"]) for sketch in batch["sketches"]]
        input_curves = [get_curves(point_input) for point_input in point_inputs]

        fig = visualize_batch(input_curves=input_curves, label_curves=label_curves,
                              sample_curves=batch["sample_curves"], box_lim=self.box_lim + 3)
        fig_path = Path(self.args.samples_dir) / f"epoch_{self.current_epoch}_batch_{batch_idx}.png"
        fig.savefig(fig_path)

    def configure_optimizers(self):
        params = list(self.model.parameters()) + list(self.mapper.parameters()) + list(self.back_mapper.parameters())
        params2 = list(self.layernorm.parameters())+list(self.post_layernorm.parameters()) + list(self.embed_patch.parameters()) +list(self.vit_mae.parameters())

        # optimizer = Adafactor(
        #         params,
        #         lr=None,
        #         eps=(1e-30, 1e-3),
        #         clip_threshold=1.0,
        #         decay_rate=-0.8,
        #         beta1=None,
        #         weight_decay=0.0,
        #         relative_step=True, #
        #         scale_parameter=True, #
        #         warmup_init=True, #
        #     )
        optimizer = optim.AdamW(params+params2, lr=self.lr)
        if not self.args.cosinedecay:
            return optimizer
            
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(self.args.epochs * 1.15), verbose=True)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=4,sefsdfsdf verbose=True)
        #lr_scheduler = AdafactorSchedule(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        }
