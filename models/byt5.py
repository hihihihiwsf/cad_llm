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
from transformers import T5Config, T5ForConditionalGeneration, AutoTokenizer
from transformers.modeling_utils import unwrap_model
import sys

sys.path.insert(0, '/home/ec2-user/SageMaker/efs/code/cad_llm')
from metrics import calculate_accuracy, calculate_first_ent_accuracy, calculate_validity
from util import get_quantized_range
from geometry.parse import get_curves, get_point_entities
from geometry.visualization import visualize_batch
from pathlib import Path
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from torch.optim.lr_scheduler import CosineAnnealingLR

import math
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

from models.modeling_t5 import T5ForConditionalGeneration
import numpy as np

class TransformerModel(nn.Module):

    def __init__(self, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        decode_layers = TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=512, dropout=dropout, batch_first=True)
        self.transformer_decoder = TransformerDecoder(decode_layers, nlayers)
        self.lmhead = nn.Linear(d_model, 64+5, bias=False)
        # self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, d_model)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        # self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def encode(self, src: Tensor, mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = src * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask=mask)
        output = self.linear(output)
        return output
    
    def decode(self, tgt, memory, tgt_mask=None):
        
        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
        return output
    
    def generate_square_subsequent_mask(self, sz, device='cuda'):
        return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class ByT5Model(pl.LightningModule):
    total_train_steps = None
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
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        self.args = args
        
        self.local_model = TransformerModel(d_model=self.model.config.d_model, nhead=4, d_hid=768, nlayers=4)
        self.initial_embedder = self.model.get_input_embeddings()
        self.lr = self.args.lr
        self.batch_size = self.args.batch_size  # to fix logging warning
        self.total_train_steps = None  # should be set later for lr scheduler

        self.quantization_bits = 6  # Hard code for now
        self.quantized_range = get_quantized_range(self.quantization_bits)
        self.box_lim = max(self.quantized_range)  # for visualization
        # self.lhead = nn.Linear(self.model.config.d_model, 64+3, bias=False)

        mapper_to_possible_outputs = {self.tokenizer.encode(str(i))[1]: i-1 for i in range(1, 65)}
        mapper_to_possible_outputs[self.tokenizer.encode(',')[1]] = 64
        mapper_to_possible_outputs[self.tokenizer.encode(';')[1]] = 64+1
        mapper_to_possible_outputs[self.tokenizer.pad_token_id] = 64+2
        mapper_to_possible_outputs[self.tokenizer.bos_token_id] = 64+3
        mapper_to_possible_outputs[self.tokenizer.eos_token_id] = 64+4
        self.token_mapper = mapper_to_possible_outputs
        
        self.back_to_llm_token_mapper = {}
        for k, v in self.token_mapper.items():
            self.back_to_llm_token_mapper[v] = k
        
        # If using single token encoding - adjust tokenizer and model embeddings
        if not args.ascii_encoding:
            self.adjust_to_use_new_tokens()

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
        cols = ["input_ids", "attention_mask", "labels"]
        model_batch = {col: val for col, val in batch.items() if col in cols}
        txt_embeddings = self.initial_embedder(batch['input_entities'].input_ids)
        #torch attention masking is oppostie of huggingface
        mask = (-(batch['input_entities'].attention_mask) + 1).float()
        txt_embeddings = self.local_model.encode(txt_embeddings, mask=mask)
        # txt_embeddings = torch.sum(txt_embeddings, 1)
        txt_embeddings = txt_embeddings[:, 0, :]
        pad_embed = self.initial_embedder(torch.tensor([self.tokenizer.pad_token_id]).to(self.device))
                
        ########### INPUT CHUNKS
        # Find the maximum chunk size
        chunks = batch['input_ent_length']
        # Create an empty tensor with zero padding
        input_tensor = pad_embed.repeat(len(chunks), max(chunks), 1)
        idx = 0
        for i, size in enumerate(chunks):
            input_tensor[i, :size, :] = txt_embeddings[idx : idx + size]
            idx += size
        
        # input_tensor = torch.zeros(len(chunks), max(chunks), 512).to(self.device)

        ########### OUTPUT CHUNKS
        with torch.no_grad():
            txt_embeddings = self.initial_embedder(batch['output_entities'].input_ids)
            mask = (-(batch['output_entities'].attention_mask) + 1).float()
            txt_embeddings = self.local_model.encode(txt_embeddings, mask=mask)
            txt_embeddings = txt_embeddings[:, 0, :]
            chunks = batch['output_ent_length']
            output_tensor = pad_embed.repeat(len(chunks), max(chunks), 1)
            idx = 0
            decoder_inputs_embeds = []
            for i, size in enumerate(chunks):
                output_tensor[i, :size, :] = txt_embeddings[idx : idx + size]
                ## adding a pad token to the begining because decoder_input_token should be shifted right
                decoder_inputs_embeds.append(torch.concatenate((pad_embed, output_tensor[i, :, :]), dim=0).unsqueeze(0))
                idx += size
            decoder_inputs_embeds = torch.cat(decoder_inputs_embeds, dim=0)
            decoder_inputs_embeds.requires_grad_(False)
        
            
        model_batch['inputs_embeds'] = input_tensor
        model_batch['decoder_inputs_embeds'] = decoder_inputs_embeds
        del model_batch['input_ids'], 
        del model_batch['labels']
        model_batch['attention_mask'] =  batch['batch_att_mask']
        outputs = self.model(**model_batch, output_hidden_states=True)
        o = batch['output_entities'].input_ids
        o = torch.concatenate((o[:, 0].unsqueeze(1), o[:, 2:]), dim=1)
        tgt = self.initial_embedder(o) # ignoring "C"
        
        l = batch['output_ent_length']
        unpacked_entities = []
        for i,j in enumerate(l):
            unpacked_entities.append(outputs["decoder_hidden_states"][-1][i, :j, :])
        unpacked_entities = torch.cat(unpacked_entities, dim=0).unsqueeze(1)
        
        outputs = self.local_model.decode(tgt, unpacked_entities, tgt_mask=nn.Transformer.generate_square_subsequent_mask(tgt.shape[1]).to(self.device))
        outputs = self.local_model.lmhead(outputs)
        idx = 0

        with torch.no_grad():
            lbl = o.clone()
            for key, value in self.token_mapper.items():
                lbl[lbl == key] = value
        
        # lbl = torch.randint(0, 69, (outputs.shape[0], outputs.shape[1]))
        
        loss = torch.nn.functional.cross_entropy(outputs.permute(0, 2, 1), lbl)
        # loss = torch.nn.functional.cross_entropy(outputs[:, 0], lbl[:, 0])    
        
        # asd = torch.argmax(outputs, 2).clone()
        # for key, value in self.back_to_llm_token_mapper.items():
        #     asd[asd == key] = value
        # self.tokenizer.decode(asd[0, :])
        
        # for i, j in enumerate(outputs):
        #     #ignoring first two tokens which are "<s>" and "C"
        #     lbl = batch['output_entities'].input_ids[idx: idx+l[i], 2:]  #entitites of the first sample in the batch
        #     lbl = lbl.reshape(1, -1).squeeze()
        #     model_output = outputs[i, :lbl.shape[0], :]
        #     lbl = list(map(lambda x:self.token_mapper[x.item()], lbl))
        #     lbl = torch.tensor(lbl).to(self.device)
        #     lbl = batch['output_entities'].input_ids[0, :].squeeze()
        #     loss += torch.nn.functional.cross_entropy(model_output, lbl)
            # print(loss)
        
        self.log("loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True,
                batch_size=self.batch_size, sync_dist=True)
        # loss = outputs.loss  # CrossEntropyLoss(ignore_index=-100) between outputs.logits and labels
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True,
                 batch_size=self.batch_size, sync_dist=True)
        return loss

    
    
    def validation_step(self, batch, batch_idx):
        cols = ["input_ids", "attention_mask", "labels"]
        model_batch = {col: val for col, val in batch.items() if col in cols}
        txt_embeddings = self.initial_embedder(batch['input_entities'].input_ids)
        #torch attention masking is oppostie of huggingface
        mask = (-(batch['input_entities'].attention_mask) + 1).float()
        txt_embeddings = self.local_model.encode(txt_embeddings, mask=mask)
        # txt_embeddings = torch.sum(txt_embeddings, 1)
        txt_embeddings = txt_embeddings[:, 0, :]
        pad_embed = self.initial_embedder(torch.tensor([self.tokenizer.pad_token_id]).to(self.device))
                
        ########### INPUT CHUNKS
        # Find the maximum chunk size
        chunks = batch['input_ent_length']
        # Create an empty tensor with zero padding
        input_tensor = pad_embed.repeat(len(chunks), max(chunks), 1)
        idx = 0
        for i, size in enumerate(chunks):
            input_tensor[i, :size, :] = txt_embeddings[idx : idx + size]
            idx += size
        
        # input_tensor = torch.zeros(len(chunks), max(chunks), 512).to(self.device)

        ########### OUTPUT CHUNKS
        with torch.no_grad():
            txt_embeddings = self.initial_embedder(batch['output_entities'].input_ids)
            mask = (-(batch['output_entities'].attention_mask) + 1).float()
            txt_embeddings = self.local_model.encode(txt_embeddings, mask=mask)
            txt_embeddings = txt_embeddings[:, 0, :]
            chunks = batch['output_ent_length']
            output_tensor = pad_embed.repeat(len(chunks), max(chunks), 1)
            idx = 0
            decoder_inputs_embeds = []
            for i, size in enumerate(chunks):
                output_tensor[i, :size, :] = txt_embeddings[idx : idx + size]
                ## adding a pad token to the begining because decoder_input_token should be shifted right
                decoder_inputs_embeds.append(torch.concatenate((pad_embed, output_tensor[i, :, :]), dim=0).unsqueeze(0))
                idx += size
            decoder_inputs_embeds = torch.cat(decoder_inputs_embeds, dim=0)
            decoder_inputs_embeds.requires_grad_(False)
        
            
        model_batch['inputs_embeds'] = input_tensor
        model_batch['decoder_inputs_embeds'] = decoder_inputs_embeds
        del model_batch['input_ids'], 
        del model_batch['labels']
        model_batch['attention_mask'] =  batch['batch_att_mask']
        outputs = self.model(**model_batch, output_hidden_states=True)
        o = batch['output_entities'].input_ids
        o = torch.concatenate((o[:, 0].unsqueeze(1), o[:, 2:]), dim=1)
        tgt = self.initial_embedder(o) # ignoring "C"
        
        l = batch['output_ent_length']
        unpacked_entities = []
        for i,j in enumerate(l):
            unpacked_entities.append(outputs["decoder_hidden_states"][-1][i, :j, :])
        unpacked_entities = torch.cat(unpacked_entities, dim=0).unsqueeze(1)
        
        outputs = self.local_model.decode(tgt, unpacked_entities, tgt_mask=nn.Transformer.generate_square_subsequent_mask(tgt.shape[1]).to(self.device))
        outputs = self.local_model.lmhead(outputs)
        idx = 0

        lbl = o.clone()
        for key, value in self.token_mapper.items():
            lbl[lbl == key] = value

            
        # lbl = torch.randint(0, 69, (outputs.shape[0], outputs.shape[1]))
        
        loss = torch.nn.functional.cross_entropy(outputs.permute(0, 2, 1), lbl)

        
        self.log("val_iter_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True,
            batch_size=self.batch_size, sync_dist=True)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=self.batch_size, sync_dist=True)
        

            
        return loss
    
    
    
    def validation_step_34(self, batch, batch_idx):
        
        cols = ["input_ids", "attention_mask", "labels"]
        model_batch = {col: val for col, val in batch.items() if col in cols}
        txt_embeddings = self.initial_embedder(batch['input_entities'].input_ids)
        #torch attention masking is oppostie of huggingface
        mask = (-(batch['input_entities'].attention_mask) + 1).float()
        txt_embeddings = self.local_model.encode(txt_embeddings, mask=mask)
        # txt_embeddings = torch.sum(txt_embeddings, 1)
        txt_embeddings = txt_embeddings[:, 0, :]
        pad_embed = self.initial_embedder(torch.tensor([self.tokenizer.pad_token_id]).to(self.device))
        start_embed = self.initial_embedder(torch.tensor([self.tokenizer.bos_token_id]).to(self.device))
        ########### INPUT CHUNKS
        # Find the maximum chunk size
        chunks = batch['input_ent_length']
        
        input_tensor = pad_embed.repeat(len(chunks), max(chunks), 1)
        idx = 0
        for i, size in enumerate(chunks):
            input_tensor[i, :size, :] = txt_embeddings[idx : idx + size]
            idx += size
            
        chunks = batch['output_ent_length']
        l = batch['output_ent_length']
        
        model_batch['inputs_embeds'] = input_tensor
        del model_batch['input_ids'], 
        del model_batch['labels']
        model_batch['attention_mask'] =  batch['batch_att_mask']
        outputs = self.model.generate(**model_batch, output_hidden_states=True, return_dict_in_generate=True, max_new_tokens=30)

        
        # txt_embeddings = self.initial_embedder(batch['output_entities'].input_ids)
        # mask = (-(batch['output_entities'].attention_mask) + 1).float()
        # txt_embeddings = self.local_model.encode(txt_embeddings, mask=mask)
        # txt_embeddings = txt_embeddings[:, 0, :]
        
        # # Create an empty tensor with zero padding
        # output_tensor = pad_embed.repeat(len(chunks), max(chunks), 1)
        # idx = 0
        # for i, size in enumerate(chunks):
        #     output_tensor[i, :size, :] = txt_embeddings[idx : idx + size]
        #     idx += size
        #decoder hidden states is a tuple of size # new_tokens * num_layers * bs, 1, h_dim
        decoder_hidden_states = []
        for e in outputs['decoder_hidden_states']:
            decoder_hidden_states.append(e[-1])
        
        
        src = torch.cat(decoder_hidden_states, dim=1)
        src = src.view(-1, 1, src.shape[2])
        
        # trimmed_src = []
        # for i, s in enumerate(src):
        #     trimmed_src.append(s[:batch['output_ent_length'][i], :].unsqueeze(0))
        # src = torch.cat(trimmed_src, dim=1)
        # target = pad_embed.repeat(src.shape[0],1, 1)
        target = start_embed.repeat(src.shape[0],1, 1)
        for i in range(18): #circle = "<s> + 8*2 + </s>"
            outputs = self.local_model.decode(target, src)
            predicted_value = outputs[:, -1, :].unsqueeze(1)
            target = torch.concatenate((target, predicted_value),dim=1)
        
        outputs = self.local_model.lmhead(target[:, 1:, :]) #ignoring the start token = <pad> or <s>
        lbl = torch.full(tuple(outputs.shape[:2]), self.token_mapper[self.tokenizer.pad_token_id])
        max_num_ent = outputs.shape[1]
        idx, step = 0, 0
        gt = batch['output_entities'].input_ids[:, 2:] #ignoring <s> and C 
        for i, j in enumerate(l):
            lbl[i*max_num_ent:i*max_num_ent+j, :gt.shape[1]] = gt[idx:idx+j]
            idx += j
        
        for key, value in self.token_mapper.items():
            lbl[lbl == key] = value
        
        
        loss = torch.nn.functional.cross_entropy(outputs.permute(0, 2, 1), lbl.to(self.device))
        final_seq = torch.argmax(outputs, 2).view(len(l), -1, outputs.shape[1])
        final_seq = final_seq.view(len(l), 1, -1).squeeze()
        for key, value in self.back_to_llm_token_mapper.items():
            final_seq[final_seq == key] = value
        batch['samples'] = final_seq
        
        
        # seq_samples = []
        # for i in final_seq:
        #     sample = list(map(lambda x:self.back_to_llm_token_mapper[x.item()], i))
        #     sample = torch.tensor(sample).to(self.device).unsqueeze(0)
        #     seq_samples.append(sample)
            
        # batch['samples'] = torch.cat(seq_samples, dim=0)
        # loss = 0
        # idx = 0

        
        # for i, j in enumerate(outputs):
        #     #ignoring first two tokens which are "<s>" and "C"
        #     lbl = batch['output_entities'].input_ids[idx: idx+l[i], 2:]  #entitites of the first sample in the batch
        #     lbl = lbl.reshape(1, -1).squeeze()
        #     model_output = outputs[i, :lbl.shape[0], :]
        #     lbl = list(map(lambda x:self.token_mapper[x.item()], lbl))
        #     lbl = torch.tensor(lbl).to(self.device)
            # loss += torch.nn.functional.cross_entropy(model_output, lbl)

        self.log("val_iter_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True,
                batch_size=self.batch_size, sync_dist=True)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=self.batch_size, sync_dist=True)

        # Generate and process samples
        self.generate_samples(batch)

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

    def generate_samples(self, batch):
        # Recursively unwrap the model from potential distributed training containers
        # generate_func = unwrap_model(self.model).generate
        # batch["samples"] = generate_func(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
        #                                  do_sample=False, max_new_tokens=self.args.max_length+10)

        string_samples = self.tokenizer.batch_decode(batch["samples"], skip_special_tokens=True)
        batch["point_samples"] = [get_point_entities(string_sample) for string_sample in string_samples]
        batch["point_labels"] = [get_point_entities(string_label) for string_label in batch["output_text"]]

        batch["sample_curves"] = [get_curves(point_sample) for point_sample in batch["point_samples"]]

    def log_samples(self, batch, batch_idx):
        label_curves = [get_curves(point_label) for point_label in batch["point_labels"]]

        batch["point_inputs"] = [get_point_entities(string_label) for string_label in batch["input_text"]]
        input_curves = [get_curves(point_input) for point_input in batch["point_inputs"]]

        fig = visualize_batch(input_curves=input_curves, label_curves=label_curves,
                              sample_curves=batch["sample_curves"], box_lim=self.box_lim + 3)
        fig_path = Path(self.args.samples_dir) / f"epoch_{self.current_epoch}_batch_{batch_idx}.png"
        fig.savefig(fig_path)

    def set_total_train_steps(self, num_train_batches):
        # Assumes running on gpus, one node and no accumulate_grad_batches
        n_gpus = torch.cuda.device_count()
        train_batches = num_train_batches // n_gpus if n_gpus else num_train_batches
        self.total_train_steps = train_batches * self.args.epochs
    
    @staticmethod
    def set_total_train_steps_ray(num_train_batches, n_gpus, epochs):
        # Assumes running on gpus, one node and no accumulate_grad_batches
        n_gpus = torch.cuda.device_count()
        train_batches = num_train_batches // n_gpus if n_gpus else num_train_batches
        ByT5Model.total_train_steps = train_batches * epochs

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
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
