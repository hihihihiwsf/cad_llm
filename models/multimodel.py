import torch
from torch import nn, einsum
import torch.nn.functional as F

from typing import Optional, Tuple, Union
from torch.nn import CrossEntropyLoss

import torch.distributed as dist

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
from transformers import T5Config, T5ForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM

from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.modeling_utils import unwrap_model
import sys

sys.path.insert(0, '/home/ec2-user/SageMaker/efs/code/cad_llm')
from metrics import calculate_accuracy, calculate_first_ent_accuracy, calculate_validity
from util import get_quantized_range
from geometry.parse import get_curves, get_point_entities
from geometry.visualization import visualize_batch, visualize_sample
from pathlib import Path
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from PIL import Image

from transformers import CLIPVisionModelWithProjection, CLIPVisionModel

class BLIPModel(nn.Module):
    def __init__(self,args):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """               
        super().__init__()
        self.args = args

        clipmodel = "facebook/dino-vitb16"
        self.visual_encoder = CLIPVisionModelWithProjection.from_pretrained(clipmodel)
        self.visual_encoder.requires_grad_(False)

        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        pretrained_codet5 = AutoModelForSeq2SeqLM.from_pretrained(args.model_name,
                                              torch_dtype=torch.float16,
                                              trust_remote_code=True)
        self.text_encoder = pretrained_codet5.encoder
        
        self.embed_dim = pretrained_codet5.config.d_model  # config.n_embd for 2b model
        self.image_proj = torch.nn.Linear(self.visual_encoder.config.projection_dim, self.embed_dim)
        self.text_proj = torch.nn.Linear(self.text_encoder.config.d_model, self.embed_dim)
        
        #self.decoder_proj = torch.nn.Linear(2*self.embed_dim, self.embed_dim)
        
        self.multimodal_decoder = pretrained_codet5.decoder

    def forward(self,            
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                images: Optional[torch.FloatTensor] = None,
                decoder_input_ids: Optional[torch.LongTensor] = None,
                decoder_attention_mask: Optional[torch.BoolTensor] = None,
                encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
                past_key_values: Tuple[Tuple[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,):
        
        # assert mode in ['image', 'text', 'multimodal'], "mode parameter must be image, text, or multimodal"

        with torch.no_grad():   
            # image features
            clip_output = self.visual_encoder(**images)  
            image_embeds = clip_output.image_embeds # shape:(bsz,clip_model_dim)
            image_embeds = self.image_proj(image_embeds)   
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long)        

        #text features
        
        text_output = self.text_encoder(input_ids = input_ids, attention_mask = attention_mask)
        text_embeds = text_output.last_hidden_state[:,0,:] #shape: (bsz, max_length, model_dim)
        

        # # return multimodel features
        # image_embeds = self.visual_encoder(image)    
        # image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long) 
        
        # batch.input_ids[:,0] = self.tokenizer.enc_token_id
        # output = self.text_encoder(batch.input_ids,
        #                             attention_mask = text.attention_mask,
        #                             encoder_hidden_states = image_embeds,
        #                             encoder_attention_mask = image_atts,      
        #                             return_dict = True,
        #                             )              
        #   return output.last_hidden_state

        concate_output = torch.concatenate(image_embeds, text_embeds)
        decoder_input = self.decoder_proj(concate_output)

        decoder_input = image_embeds+text_embeds

        decoder_outputs = self.multimodal_decoder(input_embeds = decoder_input, attention_mask = attention_mask, labels=labels)

        # Compute loss independent from decoder (as some shift the logits inside them)
        loss = None
        if labels is not None:
            # warnings.warn(DEPRECATION_WARNING, FutureWarning)
            logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.reshape(-1, self.decoder.config.vocab_size), labels.view(-1))


        if not return_dict:
            return (loss,) + decoder_outputs + encoder_outputs
        
        return Seq2SeqLMOutput(
            loss=loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            # encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            # encoder_hidden_states=encoder_outputs.hidden_states,
            # encoder_attentions=encoder_outputs.attentions,
        )

def train_VL(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=None):
    model.train()
    ddp_loss = torch.zeros(2).to(rank)
    if sampler:
        sampler.set_epoch(epoch)
    for batch_idx, batch in enumerate(train_loader):
        cols = ["input_ids", "attention_mask", "labels", "images"]
        model_batch = {col: val.to(rank) for col, val in batch.items() if col in cols}

        optimizer.zero_grad()
        output = model(**model_batch)
        loss = output.loss
        loss.backward()
        optimizer.step()

        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(batch)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, ddp_loss[0] / ddp_loss[1]))


def test_VL(model, tokenizer, args, rank, world_size, test_loader):
    model.eval()
    correct = 0
    ddp_loss = torch.zeros(3).to(rank)
    with torch.no_grad():
        for batch in test_loader:
            cols = ["input_ids", "attention_mask", "labels", "images"]
            model_batch = {col: val.to(rank) for col, val in batch.items() if col in cols}

            output = model(model_batch)
            ddp_loss[0] += output.loss.item()  # sum up batch loss
            #pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # Generate and process samples
            batch = generate_samples(model, tokenizer, args, batch)

            # Calculate metrics
            top1_full_sketch = calculate_accuracy(samples=batch["point_samples"], labels=batch["point_labels"])
            mx = 0
            for i,j in zip(batch['string_samples'], batch['string_labels']):
                i = i.strip('#')
                j = j.strip('#')
                if i == j:
                    mx += 1
            top1_full_sketch = mx/len(batch['string_labels'])

            ddp_loss[1] += top1_full_sketch #pred.eq(target.view_as(pred)).sum().item()
            ddp_loss[2] += len(batch)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    if rank == 0:
        test_loss = ddp_loss[0] / ddp_loss[2]
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, int(ddp_loss[1]), int(ddp_loss[2]),
            100. * ddp_loss[1] / ddp_loss[2]))


def generate_samples(model, tokenizer, args, batch):
        # Recursively unwrap the model from potential distributed training containers
        generate_func = unwrap_model(model).generate
        batch["samples"] = generate_func(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], max_length=args.max_length, decoder_input_ids=batch["input_ids"],
                                         do_sample=False)

        batch["string_samples"] = tokenizer.batch_decode(batch["samples"], skip_special_tokens=True)
        batch["string_labels"] = [sketch["output_text"] for sketch in batch["sketches"]]

        batch["point_samples"] = [get_point_entities(string_sample) for string_sample in batch["string_samples"]]
        batch["point_labels"] = [get_point_entities(string_label) for string_label in batch["string_labels"]]

        batch["sample_curves"] = [get_curves(point_sample) for point_sample in batch["point_samples"]]

        return batch

class VLModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()

        self.args = args
        self.model = BLIPModel(args)
        self.tokenizer = self.model.tokenizer

        self.lr = self.args.lr
        self.batch_size = self.args.batch_size  # to fix logging warning
        self.quantization_bits = 6  # Hard code for now
        self.quantized_range = get_quantized_range(self.quantization_bits)
        self.box_lim = max(self.quantized_range)  # for visualization

    def training_step(self, batch, batch_idx):
        cols = ["input_ids", "attention_mask", "labels", "images"]
        model_batch = {col: val for col, val in batch.items() if col in cols}
        
        model_batch["decoder_input_ids"] = model_batch["input_ids"].clone()
        outputs = self.model(**model_batch)
        loss = outputs.loss  # CrossEntropyLoss(ignore_index=-100) between outputs.logits and labels
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True,
                 batch_size=self.batch_size, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        cols = ["input_ids", "attention_mask", "labels", "images"]
        model_batch = {col: val for col, val in batch.items() if col in cols}
        
        model_batch["decoder_input_ids"] = model_batch["input_ids"].clone()
        outputs = self.model(**model_batch)
        loss = outputs.loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=self.batch_size, sync_dist=True)

        # Generate and process samples
        self.generate_samples(batch)

        # Calculate metrics
        top1_full_sketch = calculate_accuracy(samples=batch["point_samples"], labels=batch["point_labels"])
        mx = 0
        for i,j in zip(batch['string_samples'], batch['string_labels']):
            i = i.strip('#')
            j = j.strip('#')
            if i == j:
                mx += 1
        top1_full_sketch = mx/len(batch['string_labels'])
        self.log("top1_full_sketch", top1_full_sketch, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=self.batch_size, sync_dist=True)

        # top1_ent = calculate_first_ent_accuracy(samples=batch["point_samples"], labels=batch["point_labels"])


        mx = 0
        for i,j in zip(batch['string_samples'], batch['string_labels']):
            i = i.strip('#')
            j = j.strip('#')
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
        batch["samples"] = generate_func(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], max_length=self.args.max_length, decoder_input_ids=batch["input_ids"],
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


class BLIP_Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        clipmodel = "ViT-B/32"
        self.visual_encoder = CLIPVisionModelWithProjection.from_pretrained(clipmodel)
        self.visual_encoder.requires_grad_(False)

        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        pretrained_codet5 = AutoModelForSeq2SeqLM.from_pretrained(args.model_name,
                                              torch_dtype=torch.float16,
                                              trust_remote_code=True)
        self.text_decoder = pretrained_codet5.decoder
        
        self.mapper =torch.nn.Linear(self.visual_encoder.config.projection_dim, pretrained_codet5.get_input_embeddings().weight.shape[1])
    
    def forward(self, image, text):

        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)

        decoder_output = self.text_decoder(text.input_ids, 
                                           attention_mask = text.attention_mask,                 
                                           labels = text.labels,
                                           return_dict = True,   
                                          )   
        
