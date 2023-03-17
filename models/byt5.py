"""
ByT5 pytorch lightning model
"""


try:
    import comet_ml  # Import before torch
except ImportError:
    pass
import torch
import torch.optim as optim
import pytorch_lightning as pl
from transformers import T5Config, T5ForConditionalGeneration, AutoTokenizer
import sys  
sys.path.insert(0, '/home/ec2-user/SageMaker/efs/code/cad_llm')
import numpy as np
import pandas as pd
from util import get_quantized_range


class ByT5Model(pl.LightningModule):
    def __init__(self, model_name="google/byt5-base", checkpoint=None, no_pretrain=False, args=None):
        super().__init__()

        if no_pretrain:
            config = T5Config.from_pretrained(model_name)
            model = T5ForConditionalGeneration(config)
            model._init_weights(model)  # maybe redundant
        else:
            checkpoint = checkpoint or model_name
            model = T5ForConditionalGeneration.from_pretrained(checkpoint)

        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.args = args

        # Using new tokens is the default
        new_tokens = not (args and args.ascii_encoding)
        if new_tokens:
            self.adjust_to_use_new_tokens()

    def adjust_to_use_new_tokens(self):
        # Add new tokens to the tokenizer
        quantize_n_bits = 6  # Hard code for now
        new_tokens = [f"<{i}>" for i in get_quantized_range(quantize_n_bits)]
        self.tokenizer.add_tokens(new_tokens)

        # Add new token embeddings and initialize using learned embeddings
        self.model.resize_token_embeddings(len(self.tokenizer))
        embedding_params = self.model.get_input_embeddings().weight.data
        for i in range(len(new_tokens)):
            # start with the embedding for 'A', ensures no clash with embedding for ';'
            embedding_params[-i] = embedding_params[65 + i]

    
    def calc_metric(self, batch):

        labels = batch["labels"].to("cuda")
        all_samples = self.model.generate(input_ids=batch["input_ids"].to("cuda"),
                                attention_mask=batch["attention_mask"].to("cuda"),
                                do_sample=False,
                                max_new_tokens=labels.shape[1])

        columns=["ent_count", "missing_ent_count", "percent_hidden", "sample_ent_count", "matching_ent_count"]
        info = []

        def get_entities_from_sample(tokens):
            decoded = self.tokenizer.batch_decode([tokens], skip_special_tokens=True)[0]
            entities = [s + ';' for s in decoded.split(';') if s]
            return entities

        for i, sample in enumerate(all_samples):
            # sketch = val_dataset.get_sketch(index=i)
            sketch = batch['sketches'][i]
            i += 1
            label_ents = set([ent for j, ent in enumerate(sketch["entities"]) if not sketch["mask"][j]])
            sample_ents = set(get_entities_from_sample(sample))
            intersection = sample_ents.intersection(label_ents)
            
            info.append([
                len(sketch["entities"]), 
                len(label_ents),
                len(label_ents) / len(sketch["entities"]),
                len(sample_ents),
                len(intersection),
            ])
            
        df = pd.DataFrame(info, columns=columns)        


        df["top1_full_sketch"] = (df.matching_ent_count == df.missing_ent_count) & (df.sample_ent_count == df.missing_ent_count)


        return df


    def training_step(self, batch, batch_idx):

        cols = ["input_ids", "attention_mask", "labels"]
        model_batch = {col: val for col, val in batch.items() if col in cols}
        outputs = self.model(**model_batch)
        loss = outputs.loss  # CrossEntropyLoss(ignore_index=-100) between outputs.logits and labels
        self.log(f"train_loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss


    def validation_step(self, batch, batch_idx):
        
        cols = ["input_ids", "attention_mask", "labels"]
        model_batch = {col: val for col, val in batch.items() if col in cols}
        outputs = self.model(**model_batch)
        loss = outputs.loss
        self.log(f"val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        

        if batch_idx % 10 == 0:

            df = self.calc_metric(batch)
            self.log(f"top1_full_sketch", np.mean(df["top1_full_sketch"]), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss


    def configure_optimizers(self):
        lr = 3e-5
        optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        return optimizer