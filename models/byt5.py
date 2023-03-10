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

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss  # CrossEntropyLoss(ignore_index=-100) between outputs.logits and labels
        self.log(f"train_loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log(f"val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        lr = 3e-5
        optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        return optimizer
