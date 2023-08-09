"""
ByT5 pytorch lightning model for synthetic constraints
"""

try:
    import comet_ml  # Import before torch
except ImportError:
    pass
import pytorch_lightning as pl
import torch.optim as optim
from transformers import T5ForConditionalGeneration
from transformers.modeling_utils import unwrap_model

from preprocess.preprocess_syn_constraints import safe_constraints_from_string


class ByT5SynConstraintsModel(pl.LightningModule):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.save_hyperparameters()

        self.args = args
        self.model = None
        self.tokenizer = tokenizer
        self.lr = self.args.lr
        self.batch_size = self.args.batch_size  # to fix logging warning

    def prepare_data(self):
        # Download model
        T5ForConditionalGeneration.from_pretrained(self.args.model_name)

    def setup(self, stage):
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
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True,
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

        # Calculate metrics

        return loss

    def generate_samples(self, batch):
        # Recursively unwrap the model from potential distributed training containers
        generate_func = unwrap_model(self.model).generate
        samples = generate_func(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                do_sample=False, max_new_tokens=self.args.max_length + 10)

        samples = self.tokenizer.batch_decode(samples, skip_special_tokens=True)
        samples = [safe_constraints_from_string(sample) for sample in samples]
        return samples

    def configure_optimizers(self):
        return optim.AdamW(self.trainer.model.parameters(), lr=self.lr)
