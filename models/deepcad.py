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
from geometry.parse import get_curves, get_point_entities, get_pair_constraints
from geometry.visualization import visualize_batch
from pathlib import Path
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from PIL import Image
# import clip
import numpy as np
from transformers import CLIPVisionModelWithProjection, CLIPVisionModel, ViTMAEForPreTraining, AutoImageProcessor
from models.modeling_vlt5 import T5ForConditionalGeneration
from geometry.visualize_vit import Visualize_VIT
from geometry.visualization import visualize_sample_cv

import transformers
from transformers.optimization import Adafactor, AdafactorSchedule

from IPython import embed

import torch.nn.functional as F
from x_transformers import XTransformer

class ByT5Model(pl.LightningModule):
    def __init__(self, args, vit_mae, tokenizer,num_train_steps):
        super().__init__()
        self.save_hyperparameters()
        vocab_size = tokenizer.vocab_size

        #model = torch.nn.Transformer(d_model=256, nhead=8, num_encoder_layers=4, num_decoder_layers=4, dim_feedforward=512, dropout=0.1)
        model = XTransformer(
            dim = 256,
            enc_num_tokens = vocab_size,
            enc_depth = 4,
            enc_heads = 8,
            enc_max_seq_len = 192,
            dec_num_tokens = vocab_size,
            dec_depth = 4,
            dec_heads = 8,
            dec_max_seq_len = 192,
            tie_token_emb = True      # tie embeddings of encoder and decoder
        )
        self.num_train_steps=num_train_steps
        self.model = model
        self.tokenizer = tokenizer
        #self.model.resize_token_embeddings(len(self.tokenizer))

        
        self.embedding = torch.nn.Embedding(vocab_size, 256)

        self.args = args

        self.lr = self.args.lr
        self.batch_size = self.args.batch_size  # to fix logging warning
        self.quantization_bits = 6  # Hard code for now
        self.quantized_range = get_quantized_range(self.quantization_bits)
        self.box_lim = max(self.quantized_range)  # for visualizationx

        self.gelu = torch.nn.GELU()


    def training_step(self, batch, batch_idx):
        cols = ["input_ids", "attention_mask", "labels"]
        model_batch = {col: val for col, val in batch.items() if col in cols}
        src = self.embedding(batch['input_ids']).permute(1,0,2)
        tgt = self.embedding(batch['labels']).permute(1,0,2)
        
        mask = batch['attention_mask'][:, None, None, :]
        mask = (1-mask)
        mask = mask.expand(-1,8,192,-1)
        mask = mask.reshape(-1,192,192)

        # tgt_mask = batch['labels']!=0
        # tgt_mask = tgt_mask.repeat(8,1)
        # tgt_mask = tgt_mask.unsqueeze(2).repeat(1, 1, 192).bool()

        #outputs = self.model(src=src, tgt=tgt, mask=mask.bool()) #, tgt_mask = tgt_mask)

        loss = self.model(src=batch['input_ids'], tgt=batch['labels'], mask=batch['attention_mask'].bool())

        
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
        
        cols = ["input_ids", "attention_mask", "labels"]
        model_batch = {col: val for col, val in batch.items() if col in cols}
        src = self.embedding(batch['input_ids']).permute(1,0,2)
        tgt = self.embedding(batch['labels']).permute(1,0,2)
        
        mask = batch['attention_mask'][:, None, None, :]
        mask = (1-mask)
        mask = mask.expand(-1,8,192,-1)
        mask = mask.reshape(-1,192,192)

        # tgt_mask = batch['labels']!=0
        # tgt_mask = tgt_mask.repeat(8,1)
        # tgt_mask = tgt_mask.unsqueeze(2).repeat(1, 1, 192).bool()

        #outputs = self.model(src=src, tgt=tgt, mask=mask.bool()) #, tgt_mask = tgt_mask)

        loss = self.model(src=batch['input_ids'], tgt=batch['labels'], mask=batch['attention_mask'].bool())

        

        generate_func = unwrap_model(self.model).generate
        seq_out_start=torch.ones(batch['input_ids'].shape[0], 1).long().to(batch['input_ids'].device)

        batch["samples"] = generate_func(seq_in=batch["input_ids"], seq_out_start=seq_out_start,attn_mask=batch["attention_mask"],
                                         seq_len=self.args.max_length, eos_token=0)

        # token_ids = torch.argmax(outputs, dim=-1)
        # batch["string_samples"] = self.tokenizer.batch_decode(token_ids, skip_special_tokens=False)

        # batch["string_labels"] = [sketch["output_text"].replace ('</s>', '') for sketch in batch["sketches"]]
        # batch["point_samples"] = [get_pair_constraints(string_sample) for string_sample in batch["string_samples"]]
        # batch["point_labels"] = [get_pair_constraints(string_label) for string_label in batch["string_labels"]]

        # Generate and process samples
        self.generate_samples(batch)

        self.log(f"{validate}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=self.batch_size, sync_dist=True)
        #embed()
        # Calculate metrics
        top1_full_sketch = calculate_accuracy(samples=batch["point_samples"], labels=batch["point_labels"])

        self.log(f"{validate}_top1_full_sketch", top1_full_sketch, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=self.batch_size, sync_dist=True)

        top1_ent = calculate_first_ent_accuracy(samples=batch["point_samples"], labels=batch["point_labels"])

        self.log(f"{validate}_top1_ent", top1_ent, on_step=False, on_epoch=True, prog_bar=True, logger=True,
            batch_size=self.batch_size, sync_dist=True)

        precision, recall, f1 = calculate_f1(samples=batch["point_samples"], labels=batch["point_labels"])
        self.log(f"{validate}_f1", f1, on_step=False, on_epoch=True, prog_bar=True, logger=True,
            batch_size=self.batch_size, sync_dist=True)
        self.log(f"{validate}_precision", precision, on_step=False, on_epoch=True, prog_bar=True, logger=True,
            batch_size=self.batch_size, sync_dist=True)
        self.log(f"{validate}_recall", recall, on_step=False, on_epoch=True, prog_bar=True, logger=True,
            batch_size=self.batch_size, sync_dist=True)
        return loss


    
    def generate_samples(self, batch):

        batch["string_samples"] = self.tokenizer.batch_decode(batch["samples"], skip_special_tokens=True)
        batch["string_labels"] = [sketch["output_text"].replace ('</s>', '') for sketch in batch["sketches"]]

        batch["point_samples"] = [get_point_entities(string_sample) for string_sample in batch['string_samples']]
        batch["point_labels"] = [get_point_entities(string_label) for string_label in batch["string_labels"]]
        #batch["sample_curves"] = [get_curves(point_sample) for point_sample in batch["point_samples"]]

    def log_samples(self, batch, batch_idx):
        label_curves = [get_curves(point_label) for point_label in batch["point_labels"]]

        point_inputs = [get_point_entities(sketch["input_text"]) for sketch in batch["sketches"]]
        input_curves = [get_curves(point_input) for point_input in point_inputs]

        fig = visualize_batch(input_curves=input_curves, label_curves=label_curves,
                              sample_curves=batch["sample_curves"], box_lim=self.box_lim + 3)
        fig_path = Path(self.args.samples_dir) / f"epoch_{self.current_epoch}_batch_{batch_idx}.png"
        fig.savefig(fig_path)

    def configure_optimizers(self):

        optimizer = optim.AdamW(self.trainer.model.parameters(), lr=self.lr, weight_decay=0.05)

        if not self.args.cosinedecay:
            return optimizer
            
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_train_steps, verbose=True)
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
