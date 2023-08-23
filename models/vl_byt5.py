"""
ByT5 pytorch lightning model
"""

try:
    import comet_ml  # Import before torch
except ImportError:
    pass
import torch
import torch.optim as optim
import torch.nn as nn

# import lightning.pytorch as pl
import pytorch_lightning as pl
from transformers import T5Config, AutoTokenizer
from transformers.modeling_utils import unwrap_model
import sys
from models.vis_recon import VisRecon
sys.path.insert(0, '/home/ec2-user/SageMaker/efs/code/cad_llm')
from metrics import calculate_accuracy, calculate_first_ent_accuracy, calculate_validity
from util import get_quantized_range
from geometry.parse import get_curves, get_point_entities
from geometry.visualization import visualize_batch, visualize_sample
from pathlib import Path
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from PIL import Image
# import clip
import numpy as np
from transformers import CLIPVisionModelWithProjection, CLIPVisionModel, ViTMAEForPreTraining, AutoImageProcessor
from models.modeling_vlt5 import T5ForConditionalGeneration

from torch.optim.lr_scheduler import CosineAnnealingLR

class ByT5Model(pl.LightningModule):
    def __init__(self, args, tokenizer, total_train_steps, vit_mae=None):
        super().__init__()
        self.save_hyperparameters()

        if args.untrained_model:
            config = T5Config.from_pretrained(args.model_name)
            model = T5ForConditionalGeneration(config)
            model._init_weights(model)  # maybe redundant
        else:
            model = T5ForConditionalGeneration.from_pretrained(args.model_name)

        self.model = model
        self.tokenizer = tokenizer
        
        # self.tokenizer.add_special_tokens(["<IMAGE>"])
        
        self.args = args
        self.lr = self.args.lr
        self.batch_size = self.args.batch_size  # to fix logging warning
        self.total_train_steps = total_train_steps  # should be set later for lr scheduler
        self.quantization_bits = 6  # Hard code for now
        self.quantized_range = get_quantized_range(self.quantization_bits)
        self.box_lim = max(self.quantized_range)  # for visualization
        
        
        m = VisRecon(args=args)
        m.load_from_checkpoint('s3://cad-llm-katzm/jobs/vitmae_deepmind/checkpoints/best.ckpt')
        self.vit_mae = m.model
        del m

        if self.vit_mae is not None:
            self.vis_model = self.vit_mae
            self.vis_model.config.mask_ratio = 0.
            self.vis_model.requires_grad_(False)
            self.vitmae_preprocess = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")

        self.mapper =torch.nn.Linear(self.vis_model.config.hidden_size, self.model.get_input_embeddings().weight.shape[1])
        self.fusion_image = torch.nn.Transformer(nhead=8, num_encoder_layers=1, d_model=self.vis_model.config.hidden_size)

    def training_step(self, batch, batch_idx):
        cols = ["labels"]
        model_batch = {col: val for col, val in batch.items() if col in cols}

        
        # image_features = self.clip_model.encode_image(batch['images'])
        # batch['images'] = self.vitmae_preprocess(batch['images'], return_tensors="pt")
        oi = self.vis_model.vit.encoder(self.vis_model.patchify(batch['input_images'].pixel_values))
        input_image_features = torch.unsqueeze(torch.sum(oi['last_hidden_state'], 1), 1)       # oi = self.clip_model(**batch['images'])
        # image_features = oi.image_embeds
        # image_features = oi['pooler_output']
        retrieve_image = self.vis_model.vit.encoder(self.vis_model.patchify(batch['icl_image'].pixel_values))
        icl_image_features =  torch.unsqueeze(torch.sum(retrieve_image['last_hidden_state'], 1), 1)
        
        src = torch.cat((input_image_features, icl_image_features), 1)
        image_features = self.fusion_image.encoder(src)  #shape (bsz, 2, vis_dim 768)
        
        image_for_llm = self.mapper(image_features)
        
        
        #equence_output = image_features
        self.image_embeddings = image_features #self.model.vit.layernorm(sequence_output)
        
        #self.fulltext = [sketch["full_text"] for sketch in batch['sketches']]
        self.intext = [sketch["input_text"] for sketch in batch['sketches']]
        
        txt_embedder = self.model.get_input_embeddings()
        txt_embeddings = txt_embedder(batch['input_ids']) # size: (batch_size, seq_length, 1536)
        
        input_embed = torch.concatenate((image_for_llm, txt_embeddings), dim=1)
        # input_embed = torch.concatenate((imm, image_for_llm.unsqueeze(1), code, txt_embeddings), dim=1)
        model_batch['inputs_embeds'] = input_embed #txt_embeddings

        att = torch.ones(input_embed.shape[0], image_for_llm.shape[1]).to(self.device)
        model_batch['attention_mask'] = torch.cat((att, batch['attention_mask']), dim=1)



        outputs = self.model(**model_batch)
        loss = outputs.loss  # CrossEntropyLoss(ignore_index=-100) between outputs.logits and labels
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True,
                 batch_size=self.batch_size, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        
        cols = ["labels"]
        model_batch = {col: val for col, val in batch.items() if col in cols}

        
        # image_features = self.clip_model.encode_image(batch['images'])
        # batch['images'] = self.vitmae_preprocess(batch['images'], return_tensors="pt")
        oi = self.vis_model.vit.encoder(self.vis_model.patchify(batch['input_images'].pixel_values))
        input_image_features = torch.unsqueeze(torch.sum(oi['last_hidden_state'], 1), 1)       # oi = self.clip_model(**batch['images'])
        # image_features = oi.image_embeds
        # image_features = oi['pooler_output']
        retrieve_image = self.vis_model.vit.encoder(self.vis_model.patchify(batch['icl_image'].pixel_values))
        icl_image_features =  torch.unsqueeze(torch.sum(retrieve_image['last_hidden_state'], 1), 1)
        
        src = torch.cat((input_image_features, icl_image_features), 1)
        image_features = self.fusion_image.encoder(src)  #shape (bsz, 2, 768)
        
        image_for_llm = self.mapper(image_features)
        
        
        #equence_output = image_features
        self.image_embeddings = image_features #self.model.vit.layernorm(sequence_output)
        
        #self.fulltext = [sketch["full_text"] for sketch in batch['sketches']]
        self.intext = [sketch["input_text"] for sketch in batch['sketches']]
        
        txt_embedder = self.model.get_input_embeddings()
        txt_embeddings = txt_embedder(batch['input_ids']) # size: (batch_size, seq_length, 1536)
        
        input_embed = torch.concatenate((image_for_llm, txt_embeddings), dim=1)
        # input_embed = torch.concatenate((imm, image_for_llm.unsqueeze(1), code, txt_embeddings), dim=1)
        model_batch['inputs_embeds'] = input_embed #txt_embeddings

        # adding ones to attention_mask
        att = torch.ones(input_embed.shape[0], image_for_llm.shape[1]).to(self.device)
        model_batch['attention_mask'] = torch.cat((att, batch['attention_mask']), dim=1)

        batch['attention_mask'] = model_batch['attention_mask']
        batch['inputs_embeds'] = model_batch['inputs_embeds']

        outputs = self.model(**model_batch)
        loss = outputs.loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=self.batch_size, sync_dist=True)
        
        
        # Generate and process samples
        self.generate_samples(batch)


        # Calculate metrics
        # top1_full_sketch = calculate_accuracy(samples=batch["point_samples"], labels=batch["point_labels"])
        mx = 0
        for i,j in zip(batch['string_samples'], batch['string_labels']):
            out, l = i.split(";"), j.split(";")
            # label_all_ent = j.split(";")
            if set(out) == set(l):
                mx += 1
        top1_full_sketch = mx/len(batch['string_labels'])
        self.log("top1_full_sketch", top1_full_sketch, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=self.batch_size, sync_dist=True)

        # top1_ent = calculate_first_ent_accuracy(samples=batch["point_samples"], labels=batch["point_labels"])

        mx = 0
        for i,j in zip(batch['string_samples'], batch['string_labels']):
            label_all_ent = j.split(";")
            first_ent = i.split(";")[0]
            if first_ent in label_all_ent:
                mx += 1
        top1_ent = mx/len(batch['string_labels'])
        self.log("top1_ent", top1_ent, on_step=False, on_epoch=True, prog_bar=True, logger=True,
            batch_size=self.batch_size, sync_dist=True)
        # Convert string entities to curves and check validity
        validity = calculate_validity(batch_sample_curves=batch["sample_curves"])
        self.log("validity", validity, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=self.batch_size, sync_dist=True)


        # # Plot sketches
        if batch_idx < 5:
            self.log_samples(batch=batch, batch_idx=batch_idx)
        
        return loss

    def generate_samples(self, batch):
        # Recursively unwrap the model from potential distributed training containers
        generate_func = unwrap_model(self.model).generate
        batch["samples"] = generate_func(inputs_embeds=batch["inputs_embeds"], attention_mask=batch["attention_mask"],
                                         do_sample=False, max_new_tokens=self.args.max_length+10)

        batch["string_samples"] = self.tokenizer.batch_decode(batch["samples"], skip_special_tokens=True)
        batch["string_labels"] = [sketch["output_text"] for sketch in batch["sketches"]]

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

    @staticmethod
    def get_total_train_steps(num_train_batches, num_gpus, epochs):
        # Assumes running on gpus, one node and no accumulate_grad_batches
        train_batches = num_train_batches // num_gpus if num_gpus else num_train_batches
        total_train_steps = train_batches * epochs
        
        return total_train_steps

    @staticmethod
    def get_tokenizer(model_name):
        return AutoTokenizer.from_pretrained(model_name)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.trainer.model.parameters(), lr=self.lr)
        if not self.args.cosinedecay:
            return optimizer

        scheduler = CosineAnnealingLR(optimizer, T_max=self.total_train_steps, eta_min=self.lr * 0.1)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=4, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }
