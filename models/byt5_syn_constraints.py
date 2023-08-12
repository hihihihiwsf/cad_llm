"""
ByT5 pytorch lightning model for synthetic constraints
"""

try:
    import comet_ml  # Import before torch
except ImportError:
    pass
import json

import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.utilities import rank_zero_only
from transformers import T5ForConditionalGeneration
from transformers.modeling_utils import unwrap_model

from metrics.syn_constraints_metrics import PartIoU, PartAccuracy
from preprocess.syn_contraints_preprocess import safe_constraints_from_string, constraints_to_sets


class ByT5SynConstraintsModel(pl.LightningModule):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.save_hyperparameters()

        self.args = args
        self.model = None
        self.tokenizer = tokenizer
        self.lr = self.args.lr
        self.batch_size = self.args.batch_size  # to fix logging warning

        self.constraint_types = ["horizontal", "vertical", "parallel", "perpendicular"]
        self.iou = nn.ModuleDict({constraint_type: PartIoU() for constraint_type in self.constraint_types})
        self.acc = nn.ModuleDict({constraint_type: PartAccuracy() for constraint_type in self.constraint_types})

        self.sample_infos = []

    def prepare_data(self):
        print("in prepare_data")
        # Download model
        T5ForConditionalGeneration.from_pretrained(self.args.model_name)

    def setup(self, stage):
        print("in setup")
        self.model = T5ForConditionalGeneration.from_pretrained(self.args.model_name)
        self.tokenizer = self.tokenizer

        num_special_tokens = 3
        original_token_count = self.model.get_input_embeddings().weight.data.shape[0]
        new_token_count = len(self.tokenizer) - original_token_count

        # Add new token embeddings and initialize using learned embeddings
        self.model.resize_token_embeddings(len(self.tokenizer))
        embedding_params = self.model.get_input_embeddings().weight.data
        for i in range(new_token_count):
            embedding_params[original_token_count + i] = embedding_params[num_special_tokens + i]

    def training_step(self, batch, batch_idx):
        cols = ["input_ids", "attention_mask", "labels"]
        model_batch = {col: val for col, val in batch.items() if col in cols}
        outputs = self.model(**model_batch)
        loss = outputs.loss  # CrossEntropyLoss(ignore_index=-100) between outputs.logits and labels
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True,
                 batch_size=self.batch_size, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        cols = ["input_ids", "attention_mask", "labels"]
        model_batch = {col: val for col, val in batch.items() if col in cols}
        outputs = self.model(**model_batch)
        loss = outputs.loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=self.batch_size, sync_dist=True)

        # Generate and process samples
        samples = self.generate_samples(batch)

        preds = [safe_constraints_from_string(sample) for sample in samples]
        targets = [safe_constraints_from_string(constraints_srt) for constraints_srt in batch["output_text"]]

        # Calculate metrics
        for pred, target, sample, input_text in zip(preds, targets, samples, batch["input_text"]):
            self.sample_infos.append({
                "pred": pred,
                "true": target,
                "text_sample": sample,
                "input_text": input_text,
            })

            pred = constraints_to_sets(pred)
            target = constraints_to_sets(target)
            for constraint_type in self.constraint_types:
                self.iou[constraint_type].update(pred[constraint_type], target[constraint_type])
                self.acc[constraint_type].update(pred[constraint_type], target[constraint_type])

        # Log metrics
        for constraint_type in self.constraint_types:
            self.log(f"val_iou_{constraint_type}", self.iou[constraint_type], on_step=False, on_epoch=True,
                     prog_bar=True, logger=True, batch_size=self.batch_size, sync_dist=True)

            self.log(f"val_acc_{constraint_type}", self.acc[constraint_type], on_step=False, on_epoch=True,
                     prog_bar=True, logger=True, batch_size=self.batch_size, sync_dist=True)

        return loss

    @rank_zero_only
    def on_validation_epoch_end(self):
        path = f"{self.args.samples_dir}/samples_and_targets_epoch_{self.current_epoch}.json"
        with open(path, "w") as json_file:
            json.dump(self.sample_infos, json_file)

        # Reset sample_infos
        self.sample_infos = []

    def generate_samples(self, batch):
        # Recursively unwrap the model from potential distributed training containers
        generate_func = unwrap_model(self.model).generate
        samples = generate_func(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                do_sample=False, max_new_tokens=self.args.max_length)

        samples = self.tokenizer.batch_decode(samples, skip_special_tokens=True)
        return samples

    def configure_optimizers(self):
        return optim.AdamW(self.trainer.model.parameters(), lr=self.lr)
