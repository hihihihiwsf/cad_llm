"""
ByT5 pytorch lightning model
"""


try:
    import comet_ml  # Import before torch
except ImportError:
    pass
import torch.optim as optim
import pytorch_lightning as pl
from transformers import T5Config, T5ForConditionalGeneration, AutoTokenizer
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

        if args and args.ascii_encoding:
            # Add new tokens to the tokenizer
            quantize_n_bits = 6  # Hard code for now
            self.tokenizer.add_tokens([f"<{i}>" for i in get_quantized_range(quantize_n_bits)])

            # Add new token embeddings and initialize using learned embeddings
            model.resize_token_embeddings(len(self.tokenizer))
            embedding_params = model.get_input_embeddings().weight.data
            for i in range(64):
                # start with the embedding for 'A', ensures no clash with embedding for ';'
                embedding_params[-i] = embedding_params[65 + i]

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
