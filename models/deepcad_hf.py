"""
ByT5 pytorch lightning model
"""

try:
    import comet_ml  # Import before torch
except ImportError:
    pass
import torch
import torch.nn as nn
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

import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class ByT5Model(pl.LightningModule):
    def __init__(self, args, vit_mae, tokenizer, num_train_steps):
        super().__init__()
        self.save_hyperparameters()

        # if args.untrained_model:
        #     config = T5Config.from_pretrained(args.model_name)
        #     model = T5ForConditionalGeneration(config)
        #     model._init_weights(model)  # maybe redundant
        # else:
        #########
        #self.model = torch.nn.Transformer(d_model=256, nhead=8, num_encoder_layers=4, num_decoder_layers=4, dim_feedforward=512, dropout=0.1) #, activation=<function relu>, custom_encoder=None, custom_decoder=None, layer_norm_eps=1e-05, batch_first=False, norm_first=False, bias=True, device=None, dtype=None)
        model 
        d_model = 256
        nhead = 8
        num_encoder_layers=4
        num_decoder_layers=4
        dim_feedforward=512
        vocab_size = 64 * 100
        self.pos_encoder = PositionalEncoding(d_model, args.max_length)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)
        self.encoder = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, vocab_size)

        self.num_train_steps=num_train_steps
        
        self.tokenizer = tokenizer
        #self.model.resize_token_embeddings(len(self.tokenizer))
        
        self.args = args
        
        self.lr = self.args.lr
        self.batch_size = self.args.batch_size  # to fix logging warning
        self.quantization_bits = 6  # Hard code for now
        self.quantized_range = get_quantized_range(self.quantization_bits)
        self.box_lim = max(self.quantized_range)  # for visualizationx
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        encoded_src = self.transformer_encoder(src, src_mask)
        # Apply average pooling across the sequence length dimension
        pooled_output = torch.mean(encoded_src, dim=1)
        output = self.decoder(pooled_output)
        return output

    def training_step(self, batch, batch_idx):
        cols = ["input_ids", "attention_mask", "labels"]
        #src_mask = generate_square_subsequent_mask(batch['input_ids'].size(0)).to(batch['input_ids'].device)
        output = self.model(batch['input_ids'], batch['attention_mask'])
        # Adjust output dimensions and calculate loss
        output = output.view(-1, output.size(-1))
        labels = batch['labels'].view(-1)
        loss = F.cross_entropy(output, labels)
        
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
        
        ''' compare some sample with vitruvion'''
        '''
        device = batch['input_ids'].device
        strings =  ['0,38,16,24;0,24,0,38;0,11,0,24;21,29,32,43;'] #'63,32,63,51;53,51,63,51;0,32,0,51;46,44,46,51;46,44,53,44;53,44,53,51;53,44,53,51;0,32,63,32;'
        token_in = self.tokenizer(strings, padding=True, truncation=True, max_length=96, return_tensors="pt")
        batch['input_ids'] = token_in.input_ids.to(device)
        batch['attention_mask'] =token_in.attention_mask.to(device)
        point_input=get_point_entities(strings[0])
        list_of_img = visualize_sample_cv(point_entities=[point_input], box_lim=64 + 3)
        _images = self.vitmae_preprocess(list_of_img, return_tensors="pt")
        batch['images'] = _images.pixel_values.to(device)
        '''
        
        cols = ["input_ids", "attention_mask", "labels"]
        model_batch = {col: val for col, val in batch.items() if col in cols}
        output = self(batch['input_ids'], batch['attention_mask'])
        output = output.view(-1, output.size(-1))
        labels = batch['labels'].view(-1)
        loss = F.cross_entropy(output, labels)
        
        # Generate and process samples
        # self.generate_dc(batch)

        self.log(f"{validate}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=self.batch_size, sync_dist=True)
        
        # # Calculate metrics
        # top1_full_sketch = calculate_accuracy(samples=batch["point_samples"], labels=batch["point_labels"])

        # self.log(f"{validate}_top1_full_sketch", top1_full_sketch, on_step=False, on_epoch=True, prog_bar=True, logger=True,
        #          batch_size=self.batch_size, sync_dist=True)

        # top1_ent = calculate_first_ent_accuracy(samples=batch["point_samples"], labels=batch["point_labels"])

        # self.log(f"{validate}_top1_ent", top1_ent, on_step=False, on_epoch=True, prog_bar=True, logger=True,
        #     batch_size=self.batch_size, sync_dist=True)

        # precision, recall, f1 = calculate_f1(samples=batch["point_samples"], labels=batch["point_labels"])
        # self.log(f"{validate}_f1", f1, on_step=False, on_epoch=True, prog_bar=True, logger=True,
        #     batch_size=self.batch_size, sync_dist=True)
        # self.log(f"{validate}_precision", precision, on_step=False, on_epoch=True, prog_bar=True, logger=True,
        #     batch_size=self.batch_size, sync_dist=True)
        # self.log(f"{validate}_recall", recall, on_step=False, on_epoch=True, prog_bar=True, logger=True,
        #     batch_size=self.batch_size, sync_dist=True)
        return loss

    def generate_dc(self, batch):
        generated =[0]
        input_ids = batch['input_ids']
        src_mask = batch['attention_mask']
        for _ in range(self.args.max_length+10):
            input_ids = torch.tensor([generated], dtype=torch.long).to(self.transformer_encoder.device)
            output = self(input_ids, src_mask=src_mask[:, :len(generated), :len(generated)])
            next_token_logits = output[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).item()
            generated.append(next_token)
            if next_token == self.tokenizer.eos_token_id:
                break

        from IPython import embed; embed()
        

    
    def generate_samples(self, batch):
        # Recursively unwrap the model from potential distributed training containers
        generate_func = unwrap_model(self.model).generate
        batch["samples"] = generate_func(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                         do_sample=False, max_new_tokens=self.args.max_length+10)

        batch["string_samples"] = self.tokenizer.batch_decode(batch["samples"], skip_special_tokens=True)
        batch["string_labels"] = [sketch["output_text"].replace ('</s>', '') for sketch in batch["sketches"]]

        batch["point_samples"] = [get_pair_constraints(string_sample) for string_sample in batch["string_samples"]]
        batch["point_labels"] = [get_pair_constraints(string_label) for string_label in batch["string_labels"]]

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
        #params = list(self.model.parameters()) + list(self.mapper.parameters()) #+ list(self.vis_model.parameters())
        #params2= list(self.embed_patch.parameters()) + list(self.layernorm.parameters())+list(self.post_layernorm.parameters())
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
        optimizer = optim.AdamW(self.trainer.model.parameters(), lr=self.lr, weight_decay=0.05)
        #optimizer = Adafactor(params+params2, scale_parameter=True, relative_step=True, warmup_init=True, lr=None)

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
