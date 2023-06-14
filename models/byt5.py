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
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM
import numpy as np
# import bitsandbytes as bnb
# from accelerate import dispatch_model, infer_auto_device_map

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
            # model = T5ForConditionalGeneration.from_pretrained("t5-3b")
        #     device_map = {
        #         0: [0, 1, 2],
        #         1: [3, 4, 5, 6, 7, 8, 9],
        #         2: [10, 11, 12, 13, 14, 15, 16],
        #         3: [17, 18, 19, 20, 21, 22, 23],
        #                 }
            
        #     device_map = infer_auto_device_map(
        #     model,
        #     dtype='float16'
        # )

        #     model = dispatch_model(model, device_map=device_map)
            # model.parallelize(device_map)
            # model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name,
            #                                                 trust_remote_code=True)

        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        # model.config.decoder_start_token_id = self.tokenizer.bos_token_id
        # model.config.pad_token_id = self.tokenizer.pad_token_id
        
        self.args = args

        self.lr = self.args.lr
        self.batch_size = self.args.batch_size  # to fix logging warning
        self.quantization_bits = 6  # Hard code for now
        self.quantized_range = get_quantized_range(self.quantization_bits)
        self.box_lim = max(self.quantized_range)  # for visualization
        self.all_labels_length, self.wrong_labels_length, self.true_labels_length = [], [], [] 
        self.all_labels_token, self.wrong_labels_token, self.true_labels_token, self.diff_token = [], [], [], []
        # If using single token encoding - adjust tokenizer and model embeddings
        self.in_all, self.out_all, self.in_token, self.out_token = [], [], [], []
        
        unfrozen_list_decoder = [
                            "decoder.transformer.h.31.ln_1.weight",
                            "decoder.transformer.h.31.ln_1.bias",
                            "decoder.transformer.h.31.attn.qkv_proj.weight",
                            "decoder.transformer.h.31.attn.out_proj.weight",
                            "decoder.transformer.h.31.mlp.fc_in.weight",
                            "decoder.transformer.h.31.mlp.fc_in.bias",
                            "decoder.transformer.h.31.mlp.fc_out.weight",
                            "decoder.transformer.h.31.mlp.fc_out.bias",
                            "decoder.transformer.h.31.crossattention.qkv_proj.weight",
                            "decoder.transformer.h.31.crossattention.q_attn.weight",
                            "decoder.transformer.h.31.crossattention.out_proj.weight",
                            "decoder.transformer.ln_f.weight",
                            "decoder.transformer.ln_f.bias",
                            "decoder.lm_head.weight",
                            "decoder.lm_head.bias",
                            "enc_to_dec_proj.weight",
                            "enc_to_dec_proj.bias"
                        ]
        
        unfrozen_list_encoder = [
                                    "encoder.h.19.ln_1.bias",
                                    "encoder.h.19.attn.qkv_proj.weight",
                                    "encoder.h.19.attn.out_proj.weight",
                                    "encoder.h.19.mlp.fc_in.weight",
                                    "encoder.h.19.mlp.fc_in.bias",
                                    "encoder.h.19.mlp.fc_out.weight",
                                    "encoder.h.19.mlp.fc_out.bias",
                                    "encoder.ln_f.weight",
                                    "encoder.ln_f.bias"
                                ]


        # for n, p in self.model.named_parameters():
        #     if n in unfrozen_list_decoder:
        #         p.requires_grad = True
        #     else:
        #         p.requires_grad = False
            
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


    # def adjust_to_use_new_tokens(self):
    #     # Add new tokens to the tokenizer

    #     new_tokens = [f"<{i}>" for i in self.quantized_range]
    #     self.tokenizer.add_tokens(new_tokens)

    #     # Add new token embeddings and initialize using learned embeddings
    #     self.model.resize_token_embeddings(len(self.tokenizer))
    #     embedding_params = self.model.get_input_embeddings().weight.data

    #     neg, pos = [], []
    #     total = 0
    #     for i, j in enumerate(self.quantized_range):
    #         if j != 0:
    #             #assigning the negative of the absolute value of that token
    #             l = len(self.quantized_range)
    #             embedding_params[-l + i] = np.sign(j) * embedding_params[self.tokenizer.encode(str(abs(j)))[0]]
    #         else:
    #             embedding_params[-l + i] = torch.zeros_like(embedding_params[0])


    def training_step(self, batch, batch_idx):
        cols = ["input_ids", "attention_mask", "labels"]
        model_batch = {col: val for col, val in batch.items() if col in cols}
        # model_batch['decoder_input_ids'] = model_batch['input_ids'].clone()
        
        for s in batch['sketches']:
            self.in_token.append(len(self.tokenizer.tokenize(s['input_text'])))
            self.out_token.append(len(self.tokenizer.tokenize(s['output_text'])))
            self.in_all.append(s['input_ent_num'])
            self.out_all.append(s['output_ent_num'])
        
        outputs = self.model(**model_batch)
        loss = outputs.loss  # CrossEntropyLoss(ignore_index=-100) between outputs.logits and labels
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True,
                 batch_size=self.batch_size, sync_dist=True)
        return loss

    
    def on_train_epoch_end_reserve(self):
        import matplotlib.pyplot as plt
        
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))

        axs[0, 0].hist(self.in_all, color='blue', bins=35)
        axs[0, 0].set_title('Input Entity')

        axs[0, 1].hist(self.out_all, color='red', bins=35)
        axs[0, 1].set_title('Output Entity')


        axs[1, 0].hist(self.in_token, color='blue', bins=350)
        axs[1, 0].set_title('Input Token')

        axs[1, 1].hist(self.out_token, color='red', bins=350)
        axs[1, 1].set_title('Output Token')
        
        if self.in_all == self.out_all:
            print('all ridim')
        if self.in_token == self.out_token:
            print('token ridim')
        plt.tight_layout()
        plt.savefig('all.png')

        return
        
    
    def validation_step(self, batch, batch_idx):
        cols = ["input_ids", "attention_mask", "labels"]
        model_batch = {col: val for col, val in batch.items() if col in cols}
        
        # model_batch['decoder_input_ids'] = model_batch['input_ids'].clone()
        # batch['decoder_input_ids'] = batch['input_ids'].clone()
        
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
            self.all_labels_length.append(len(l)-1)
            self.all_labels_token.append(len(self.tokenizer.tokenize(j)))
            if set(out) == set(l):
                mx += 1
            else:
                self.wrong_labels_length.append(len(out)-1)
                self.true_labels_length.append(len(l)-1)
                
                self.wrong_labels_token.append(len(self.tokenizer.tokenize(i)))
                self.true_labels_token.append(len(self.tokenizer.tokenize(j)))
                self.diff_token.append(len(self.tokenizer.tokenize(i)) - len(self.tokenizer.tokenize(j)))

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
        # if batch_idx < 5:
        #     self.log_samples(batch=batch, batch_idx=batch_idx)

        return loss
    
    def on_validation_epoch_end_reserve(self):
        import matplotlib.pyplot as plt
        plt.hist(self.all_labels_length, color='blue')
        plt.hist(self.wrong_labels_length, alpha=.4, color='r')
        plt.savefig('wrong_vs_all.png')
        
        plt.close()
        plt.hist(self.true_labels_length, color='blue')
        plt.hist(self.wrong_labels_length, alpha=.4, color='r')
        plt.savefig('wrong_vs_true.png')
        
        plt.close()
        plt.hist(self.all_labels_token, color='blue')
        plt.hist(self.wrong_labels_token, alpha=.4, color='r')
        plt.savefig('wrong_vs_all_token.png')
        
        plt.close()
        plt.hist(self.true_labels_token, color='blue')
        plt.hist(self.wrong_labels_token, alpha=.4, color='r')
        plt.savefig('wrong_vs_true_token.png')
        
        plt.close()
        plt.hist(self.diff_token, color='blue')
        plt.savefig('diff_token.png')
        
        return self.all_labels_length, self.wrong_labels_length

    def generate_samples(self, batch):
        # Recursively unwrap the model from potential distributed training containers
        generate_func = unwrap_model(self.model).generate
        batch["samples"] = generate_func(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
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

    def configure_optimizers(self):
        
        
        # return bnb.optim.Adam8bit(self.parameters(), lr=self.lr, betas=(0.9, 0.995))
        
        
        optimizer = optim.AdamW(self.trainer.model.parameters(), lr=self.lr)
        if not self.args.cosinedecay:
            return optimizer
            
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(self.args.epochs * 1.1), verbose=True)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=4, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        }