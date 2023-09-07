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
from transformers import CLIPVisionModelWithProjection, CLIPVisionModel, ViTMAEForPreTraining, AutoImageProcessor
from models.modeling_vlt5 import T5ForConditionalGeneration
from geometry.visualize_vit import Visualize_VIT

from transformers.optimization import Adafactor, AdafactorSchedule

class ByT5Model(pl.LightningModule):
    def __init__(self, args, vit_mae):
        super().__init__()
        self.save_hyperparameters()

        # if args.untrained_model:
        #     config = T5Config.from_pretrained(args.model_name)
        #     model = T5ForConditionalGeneration(config)
        #     model._init_weights(model)  # maybe redundant
        # else:
        model = T5ForConditionalGeneration.from_pretrained(args.model_name)

        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        # self.tokenizer.add_special_tokens(["<IMAGE>"])

        self.args = args

        self.lr = self.args.lr
        self.batch_size = self.args.batch_size  # to fix logging warning
        self.quantization_bits = 6  # Hard code for now
        self.quantized_range = get_quantized_range(self.quantization_bits)
        self.box_lim = max(self.quantized_range)  # for visualization
        
        
        # m = VisRecon(args=args)
        # #m.load_from_checkpoint('/home/ec2-user/results/sifan_mae/checkpoints/best.ckpt')   #patch 32: sifan-mae-ps-32-scratch-07-04-23-2320/      vitmae_deepmind/   
        # m.load_from_checkpoint('s3://cad-llm-katzm/jobs/vitmae_deepmind/checkpoints/best.ckpt')
        # self.vit_mae = m.model 
        # self.vit_mae.requires_grad_(False)
        

        self.vit_mae = vit_mae
        if vit_mae is not None:
            self.vit_mae = vit_mae
        else:
            m = VisRecon(args=args)
            #m.load_from_checkpoint('s3://cad-llm-katzm/jobs/vitmae_deepmind/checkpoints/best.ckpt')
            m.load_from_checkpoint('s3://cad-llm-katzm/checkpoints/vitmae_sg/best.ckpt')
            self.vit_mae = m.model 
        
        self.vis_model = self.vit_mae
        self.vis_model.config.mask_ratio = 0.
        self.vis_model.requires_grad_(False)
        self.vitmae_preprocess = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
       
        
        # self.mapper =torch.nn.Linear(self.clip_model.visual.output_dim, self.model.get_input_embeddings().weight.shape[1])
        # self.mapper =torch.nn.Linear(self.clip_model.config.projection_dim, self.model.get_input_embeddings().weight.shape[1])
        # self.mapper =torch.nn.Linear(self.clip_model.config.hidden_size, self.model.get_input_embeddings().weight.shape[1])
        self.mapper =torch.nn.Linear(self.vis_model.config.hidden_size, self.model.get_input_embeddings().weight.shape[1])

        self.post_layernorm = torch.nn.LayerNorm(self.vis_model.config.hidden_size, eps=1e-5)
        self.layernorm = torch.nn.LayerNorm(self.model.get_input_embeddings().weight.shape[1], eps=1e-5)

        self.patch_num = int(self.vis_model.config.image_size/self.vis_model.config.patch_size)

        self.embed_patch = torch.nn.Linear(self.patch_num*self.patch_num, self.patch_num)
        #self.conv = torch.nn.Conv1d(self.patch_num*self.patch_num, self.patch_num, kernel_size=3, padding='same')
        #self.bn = torch.nn.BatchNorm1d(self.vis_model.config.hidden_size)
        #self.lkrelu = torch.nn.LeakyReLU()
        self.gelu = torch.nn.GELU()


    def training_step(self, batch, batch_idx):
        cols = ["attention_mask", "labels"]
        model_batch = {col: val for col, val in batch.items() if col in cols}

        #convert to PIL image for CLIP
        # img = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        with torch.no_grad():
            oi = self.vis_model.vit.encoder(self.vis_model.patchify(batch['images']))
            #image_features = torch.sum(oi['last_hidden_state'], 1)


        '''owl vit'''
        last_hidden_state = oi['last_hidden_state']
        #pooled_output= last_hidden_state[:, 0, :]
        image_embeds = self.post_layernorm(last_hidden_state)
        
        #image_embeds = image_embeds.permute(0,2,1)
        image_embeds = self.gelu(self.embed_patch(image_embeds.permute(0,2,1)).permute(0,2,1))
        #image_embeds = self.lkrelu(self.bn(self.conv(image_embeds).permute(0,2,1)).permute(0,2,1))
        
        image_for_llm = self.gelu(self.mapper(image_embeds.float()))
        image_for_llm = self.layernorm(image_for_llm)

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
        
        outputs = self.model(**model_batch)
        loss = outputs.loss  # CrossEntropyLoss(ignore_index=-100) between outputs.logits and labels
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True,
                 batch_size=self.batch_size, sync_dist=True)
        
        # '''measure training completely'''
        # self.generate_samples(batch)
        # # Calculate metrics
        # top1_full_sketch = calculate_accuracy(samples=batch["point_samples"], labels=batch["point_labels"])

        # self.log("train_top1_full_sketch", top1_full_sketch, on_step=False, on_epoch=True, prog_bar=True, logger=True,
        #          batch_size=self.batch_size, sync_dist=True)

        # top1_ent = calculate_first_ent_accuracy(samples=batch["point_samples"], labels=batch["point_labels"])

        # self.log("train_top1_ent", top1_ent, on_step=False, on_epoch=True, prog_bar=True, logger=True,
        #     batch_size=self.batch_size, sync_dist=True)
        # # # Convert string entities to curves and check validity
        # validity = calculate_validity(batch_sample_curves=batch["sample_curves"])
        # self.log("train_validity", validity, on_step=False, on_epoch=True, prog_bar=True, logger=True,
        #          batch_size=self.batch_size, sync_dist=True)

        # f1 = calculate_f1(samples=batch["point_samples"], labels=batch["point_labels"])
        # self.log("train_f1", f1, on_step=False, on_epoch=True, prog_bar=True, logger=True,
        #     batch_size=self.batch_size, sync_dist=True)
        
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
        oi = self.vis_model.vit.encoder(self.vis_model.patchify(batch['images'])) 
        image_features = torch.sum(oi['last_hidden_state'], 1)        # oi = self.clip_model(**batch['images'])
        # image_features = oi.image_embeds
        # image_features = oi['pooler_output']

        '''owl vit'''
        last_hidden_state = oi['last_hidden_state']
        #pooled_output= last_hidden_state[:, 0, :]
        image_embeds = self.post_layernorm(last_hidden_state)
        
        #image_embeds = image_embeds.permute(0,2,1)
        image_embeds = self.gelu(self.embed_patch(image_embeds.permute(0,2,1)).permute(0,2,1))
        #image_embeds = self.lkrelu(self.bn(self.conv(image_embeds).permute(0,2,1)).permute(0,2,1))

        image_for_llm = self.gelu(self.mapper(image_embeds.float()))
        image_for_llm = self.layernorm(image_for_llm)

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

        outputs = self.model(**model_batch)
        loss = outputs.loss
        self.log(f"{validate}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=self.batch_size, sync_dist=True)
        
        
        # Generate and process samples
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

    def log_samples(self, batch, batch_idx):
        label_curves = [get_curves(point_label) for point_label in batch["point_labels"]]

        point_inputs = [get_point_entities(sketch["input_text"]) for sketch in batch["sketches"]]
        input_curves = [get_curves(point_input) for point_input in point_inputs]

        fig = visualize_batch(input_curves=input_curves, label_curves=label_curves,
                              sample_curves=batch["sample_curves"], box_lim=self.box_lim + 3)
        fig_path = Path(self.args.samples_dir) / f"epoch_{self.current_epoch}_batch_{batch_idx}.png"
        fig.savefig(fig_path)

    def configure_optimizers(self):
        params = list(self.model.parameters()) + list(self.mapper.parameters()) #+ list(self.vis_model.parameters())
        params2= list(self.embed_patch.parameters()) + list(self.layernorm.parameters())+list(self.post_layernorm.parameters())
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
        #optimizer = optim.AdamW(params+params2, lr=self.lr, weight_decay=0.05)
        optimizer = Adafactor(params+params2, scale_parameter=True, relative_step=True, warmup_init=True, lr=None)

        if not self.args.cosinedecay:
            return optimizer
            
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(self.args.epochs * 1.15), verbose=True)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=4,sefsdfsdf verbose=True)
        lr_scheduler = AdafactorSchedule(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        }
