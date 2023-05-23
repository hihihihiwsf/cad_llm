"""
ByT5 pytorch lightning model
"""

try:
    import comet_ml  # Import before torch
except ImportError:
    pass
from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.optim as optim

import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import T5Config, T5ForConditionalGeneration, AutoTokenizer
from transformers.modeling_utils import unwrap_model
import sys

import sys 
sys.path.append("..") 
import dataset.sg_dataset as dataset

sys.path.insert(0, '/home/ec2-user/SageMaker/efs/code/cad_llm')
from metrics import calculate_accuracy, calculate_first_ent_accuracy, calculate_validity
from util import get_quantized_range
from geometry.parse import get_curves, get_point_entities
from geometry.visualization import visualize_batch
from pathlib import Path
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType


class Add_Embed(nn.Module):
    def __init__(self):
        super(Add_Embed, self).__init__()

        self.num_val_embeddings = 64 + 6
        self.embed_dim = 512*3
        # Value embeddings
        self.num_bins = 64
        self.max_entities = 16

        num_val_embeddings = len(dataset.Token) + self.num_bins

        self.val_embed = nn.Embedding(num_val_embeddings, self.embed_dim,
            padding_idx=dataset.Token.Pad)
        
        # Coordinate embeddings
        num_coord_embeddings = 2 + sum(
            [len(coord_map) for coord_map in dataset.COORD_TOKEN_MAP.values()])
        self.coord_embed = nn.Embedding(num_coord_embeddings, self.embed_dim,
            padding_idx=dataset.Token.Pad)
        
        # Position embeddings
        num_pos_embeddings = 3 + self.max_entities
        self.pos_embed = nn.Embedding(num_pos_embeddings, self.embed_dim,
            padding_idx=dataset.Token.Pad)
        # Also create output layer
        self.out = nn.Linear(self.embed_dim, num_val_embeddings) #512, 70
    
    def forward(self, model_batch):
        # cols = ["input_ids", "pos_ids", "coord_ids", "attention_mask", "labels"]
        # model_batch = {col: val for col, val in batch.items() if col in cols}
        val_embeddings = self.val_embed(model_batch['input_ids'])
        coord_embeddings = self.coord_embed(model_batch['coord_ids'])
        pos_embeddings = self.pos_embed(model_batch['pos_ids'])
        embeddings = val_embeddings+ coord_embeddings + pos_embeddings
        
        return embeddings




class ByT5Model(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()

        if args.untrained_model:
            config = T5Config.from_pretrained(args.model_name)
            model = T5ForConditionalGeneration(config)
            model._init_weights(model)  # maybe redundant
        else:
            model = T5ForConditionalGeneration.from_pretrained(args.model_name)

        self.model = model
        self.embed = Add_Embed()

        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if args.linear_decode:
            # self.num_val_embeddings = 64 + 6
            # self.embed_dim = 512
            # (self.val_embed, self.coord_embed, self.pos_embed), self.out = self.create_embedding()
            
            self.model.shared = self.embed
            self.model.lm_head = self.embed.out

        self.args = args

        self.lr = self.args.lr
        self.batch_size = self.args.batch_size  # to fix logging warning
        self.quantization_bits = 6  # Hard code for now
        self.quantized_range = get_quantized_range(self.quantization_bits)
        self.box_lim = max(self.quantized_range)  # for visualization
        
        self.avg_token_len = np.zeros((args.max_length))
        self.avg_token_loss = np.zeros((args.max_length))
        # If using single token encoding - adjust tokenizer and model embeddings
        # if not args.ascii_encoding:
        #     self.adjust_to_use_new_tokens()

        if args.lora:
            self.add_lora()


    def add_lora(self):
        lora_config = LoraConfig(
            r=16,
            lora_alpha=64,
            target_modules=["q", "v", "SelfAttention.k", "EncDecAttention.k", "SelfAttention.o", "EncDecAttention.o"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )
        # prepare int-8 model for training
        self.model = prepare_model_for_int8_training(self.model)
        # add LoRA adaptor
        self.model = get_peft_model(self.model, lora_config)
        # unfreeze embeddings
        self.model.get_input_embeddings().weight.requires_grad = True
        # unfreeze last layer
        for name, param in self.model.named_parameters():
            if "decoder.block.5" in name or name in ["decoder.final_layer_norm.weight", "lm_head.weight"]:
                param.requires_grad = True

        self.model.print_trainable_parameters()

    def adjust_to_use_new_tokens(self):
        # Add new tokens to the tokenizer

        new_tokens = [f"<{i}>" for i in self.quantized_range]
        
        self.tokenizer.add_tokens(new_tokens)

        # Add new token embeddings and initialize using learned embeddings
        self.model.resize_token_embeddings(len(self.tokenizer))
        embedding_params = self.model.get_input_embeddings().weight.data
        for i in range(1, len(new_tokens)+1):
            # start with the embedding for 'A', ensures no clash with embedding for ';'
            embedding_params[-i] = embedding_params[67 + i]

    def training_step(self, batch, batch_idx):
        # cols = ["input_ids", "attention_mask", "labels"]
        # model_batch = {col: val for col, val in batch.items() if col in cols}
        # outputs = self.model(**model_batch)

        cols = ["input_ids", "pos_ids", "coord_ids", "attention_mask", "labels"]
        model_batch = {col: val for col, val in batch.items() if col in cols}
        outputs = self.model(**model_batch, output_hidden_states=True)
        decoder_output = outputs.decoder_hidden_states[-1]
        loss = outputs.loss  # CrossEntropyLoss(ignore_index=-100) between outputs.logits and labels
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True,
                 batch_size=self.batch_size, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        #cols = ["input_ids", "attention_mask", "labels"]

        cols = ["input_ids", "pos_ids", "coord_ids", "attention_mask", "labels"]
        model_batch = {col: val for col, val in batch.items() if col in cols}
        outputs = self.model(**model_batch)
        loss = outputs.loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=self.batch_size, sync_dist=True)
    
        labels = model_batch['labels']
        labels[labels==-100] = 0
        if labels.shape[1]<96:
            self.avg_token_len += np.pad(np.array(torch.count_nonzero(labels, dim=0).cpu()), (0, self.args.max_length-labels.shape[1]), 'constant')
        else:
            self.avg_token_len += np.array(torch.count_nonzero(labels, dim=0).cpu())
        
        # Average loss of each token of each bath, avg_token_loss.shape [31]

        # for token_id in range(outputs.logits.shape[1]):
        #     self.avg_token_loss[token_id] = F.cross_entropy(outputs.logits.transpose(2,1)[:,:,token_id], model_batch['labels'][:,token_id],ignore_index=-100)
        #     self.log(f"avg_token_loss_token_{token_id}", self.avg_token_loss[token_id], on_step=False, on_epoch=True, prog_bar=True, logger=True,
        #             batch_size=self.batch_size, sync_dist=True)
        #     self.log(f"length_of_token_{token_id}", self.avg_token_len[token_id], on_step=False, on_epoch=True, prog_bar=True, logger=True,
        #             batch_size=self.batch_size, sync_dist=True)

        # Generate and process samples
        self.generate_samples(batch)
        logits = outputs.logits
        loss = outputs.loss
        # outputs = self.linear_decode(logits)

        # Calculate metrics
        top1_full_sketch = calculate_accuracy(samples=batch["point_samples"], labels=batch["point_labels"])
        self.log("top1_full_sketch", top1_full_sketch, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=self.batch_size, sync_dist=True)

        top1_ent = calculate_first_ent_accuracy(samples=batch["point_samples"], labels=batch["point_labels"])
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
    
    
    # def on_validation_epoch_end(self):
    #     np.save('avg_token_loss.npy', self.avg_token_loss)
    #     np.save('avg_token_len.npy', self.avg_token_len)

    def generate_samples(self, batch):
        # Recursively unwrap the model from potential distributed training containers
        generate_func = unwrap_model(self.model).generate
        batch["samples"] = generate_func(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                         do_sample=False)

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

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        if not self.args.cosinedecay:
            return optimizer
            
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs, verbose=True)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=4, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        }
