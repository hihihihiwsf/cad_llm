"""

ByT5 v2
- Lightning wrapper for huggingface Byt5 model
- Compatible with Ray
- Does not calculate metrics
- Logs all samples every validation epoch instead

"""

try:
    import comet_ml  # Import before torch
except ImportError:
    pass

import json
from pathlib import Path

import pytorch_lightning as pl
import torch.optim as optim
from adsk_ailab_ray.tools.aws import aws_s3_sync
from transformers import T5ForConditionalGeneration
from transformers.modeling_utils import unwrap_model
from eval_metrics.top1_full_sketch_metric import Top1FullSketchMetric
from eval_metrics.top1_entity_metric import Top1EntityMetric
from eval_metrics.validity_metric import ValidityMetric
from torch.nn import ModuleDict



class ByT5v2(pl.LightningModule):
    def __init__(self, model_name, lr, batch_size, max_length, tokenizer, local_samples_path,
                 remote_samples_path, val_names):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.batch_size = batch_size  # to fix logging warning
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.local_samples_path = local_samples_path
        self.remote_samples_path = remote_samples_path
        self.val_names = val_names

        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.adjust_model_to_tokenizer()

        self._reset_sample_infos()

        self.val_name_to_full_sketch_metric = ModuleDict({name: Top1FullSketchMetric() for name in self.val_names})
        self.val_name_to_top1_entity_metric = ModuleDict({name: Top1EntityMetric() for name in self.val_names})
        self.val_name_to_validity_metric = ModuleDict({name: ValidityMetric() for name in self.val_names})

    def setup(self, stage):
        self.local_samples_path = Path(self.local_samples_path) / f"rank_{self.global_rank}/"
        self.local_samples_path.mkdir(exist_ok=True, parents=True)
        self.remote_samples_path = self.remote_samples_path + f"/rank_{self.global_rank}/"

    def adjust_model_to_tokenizer(self):
        num_special_tokens = 3
        original_token_count = self.model.get_input_embeddings().weight.data.shape[0]
        new_token_count = len(self.tokenizer) - original_token_count

        if not new_token_count:
            return

        # Add new token embeddings and initialize using learned embeddings
        self.model.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=64)
        embedding_params = self.model.get_input_embeddings().weight.data
        for i in range(new_token_count):
            embedding_params[original_token_count + i] = embedding_params[num_special_tokens + i]

    def training_step(self, batch, batch_idx):
        outputs = self.model(**self._get_model_batch(batch))
        loss = outputs.loss  # CrossEntropyLoss(ignore_index=-100) between outputs.logits and labels
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True,
                 batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        val_name = self.val_names[dataloader_idx]

        if val_name == "val":
            outputs = self.model(**self._get_model_batch(batch))
            loss = outputs.loss

            self.log(f"val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                     batch_size=self.batch_size, add_dataloader_idx=False)

        # Generate samples for all validation sets
        # Recursively unwrap the model from potential distributed training containers
        generate_func = unwrap_model(self.model).generate
        pred_tokens = generate_func(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                    do_sample=False, max_new_tokens=self.max_length+10)
        batch_pred = self.tokenizer.batch_decode_to_entities(pred_tokens, skip_special_tokens=True)
        batch_true = batch["output_entities"]

        self._update_metrics(val_name, batch_pred, batch_true)

        # Log all samples for later metric extraction
        for i in range(len(batch_pred)):
            self.sample_infos[val_name].append({
                "true": batch_true[i],
                "pred": batch_pred[i],
                "prompt": batch["input_entities"][i],
                "name": batch["name"][i],
            })

    def on_validation_epoch_end(self):
        path = self.local_samples_path / f"samples_epoch_{self.current_epoch}_rank_{self.global_rank}.json"
        print(f"Saving samples to {path} ({self.global_rank})")
        with open(path, "w") as json_file:
            json.dump(self.sample_infos, json_file)
        print("Saved")

        aws_s3_sync(self.local_samples_path, self.remote_samples_path)

        self._reset_sample_infos()

        for val_name in self.val_names:
            self.log_metric(name=f"{val_name}_top1_full_sketch", metric=self.val_name_to_full_sketch_metric[val_name])
            self.log_metric(name=f"{val_name}_top1_entity", metric=self.val_name_to_top1_entity_metric[val_name])
            self.log_metric(name=f"{val_name}_validity", metric=self.val_name_to_validity_metric[val_name])

    def _reset_sample_infos(self):
        self.sample_infos = {val_name: [] for val_name in self.val_names}

    def _update_metrics(self, val_name, batch_pred, batch_true):
        for pred, true in zip(batch_pred, batch_true):
            pred = set(tuple(sorted([tuple(p) for p in ent])) for ent in pred if ent)
            true = set(tuple(sorted([tuple(p) for p in ent])) for ent in true if ent)

            self.val_name_to_full_sketch_metric[val_name].update(pred, true)
            self.val_name_to_top1_entity_metric[val_name].update(pred, true)
            self.val_name_to_validity_metric[val_name].update(pred)

    def log_metric(self, name, metric):
        self.log(name, metric.compute(), on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 add_dataloader_idx=False)

    def _get_model_batch(self, batch):
        cols = ["input_ids", "attention_mask", "labels"]
        return {col: val for col, val in batch.items() if col in cols}

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.trainer.model.parameters(), lr=self.lr)
        return optimizer
