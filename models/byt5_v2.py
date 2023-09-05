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


class ByT5v2(pl.LightningModule):
    def __init__(self, model_name, lr, batch_size, max_length, tokenizer_length, local_samples_path,
                 remote_samples_path):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.batch_size = batch_size  # to fix logging warning
        self.max_length = max_length
        self.local_samples_path = local_samples_path
        self.remote_samples_path = remote_samples_path
        self.tokenizer_length = tokenizer_length
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.adjust_model_to_tokenizer()

        self.sample_infos = []

    def setup(self, stage):
        self.local_samples_path = Path(self.local_samples_path) / f"rank_{self.global_rank}/"
        self.local_samples_path.mkdir(exist_ok=True, parents=True)
        self.remote_samples_path = self.remote_samples_path + f"/rank_{self.global_rank}/"

    def adjust_model_to_tokenizer(self):
        num_special_tokens = 3
        original_token_count = self.model.get_input_embeddings().weight.data.shape[0]
        new_token_count = self.tokenizer_length - original_token_count

        if not new_token_count:
            return

        # Add new token embeddings and initialize using learned embeddings
        self.model.resize_token_embeddings(self.tokenizer_length)
        embedding_params = self.model.get_input_embeddings().weight.data
        for i in range(new_token_count):
            embedding_params[original_token_count + i] = embedding_params[num_special_tokens + i]

    def training_step(self, batch, batch_idx):
        outputs = self.model(**self._get_model_batch(batch))
        loss = outputs.loss  # CrossEntropyLoss(ignore_index=-100) between outputs.logits and labels
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True,
                 batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(**self._get_model_batch(batch))
        loss = outputs.loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=self.batch_size)

        # Generate and process samples
        # Recursively unwrap the model from potential distributed training containers
        generate_func = unwrap_model(self.model).generate
        samples = generate_func(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                do_sample=False, max_new_tokens=self.max_length+10)

        # Log all samples for later metric extraction
        for i in range(len(samples)):
            self.sample_infos.append({
                "samples": samples[i].tolist(),
                "input_ids": batch["input_ids"][i].tolist(),
                "labels": batch["labels"][i].tolist(),
                "input_text": batch["input_text"][i],
                "output_text": batch["output_text"][i],
                "name": batch["name"][i],
            })

        return loss

    def on_validation_epoch_end(self):

        path = self.local_samples_path / f"samples_epoch_{self.current_epoch}_rank_{self.global_rank}.json"
        print(f"Saving samples to {path} ({self.global_rank})")
        with open(path, "w") as json_file:
            json.dump(self.sample_infos, json_file)
        print("Saved")

        aws_s3_sync(self.local_samples_path, self.remote_samples_path)

        # Reset sample_infos
        self.sample_infos = []

    def _get_model_batch(self, batch):
        cols = ["input_ids", "attention_mask", "labels"]
        return {col: val for col, val in batch.items() if col in cols}

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.trainer.model.parameters(), lr=self.lr)
        return optimizer
