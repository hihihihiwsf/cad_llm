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
from metrics import calculate_accuracy
from util import get_quantized_range


class ByT5Model(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        if args.untrained_model:
            config = T5Config.from_pretrained(args.model_name)
            model = T5ForConditionalGeneration(config)
            model._init_weights(model)  # maybe redundant
        else:
            model = T5ForConditionalGeneration.from_pretrained(args.model_name)

        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        self.args = args

        # If using single token encoding - adjust tokenizer and model embeddings
        if not args.ascii_encoding:
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
            samples = self.model.generate(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                          do_sample=False, max_new_tokens=batch["labels"].shape[1])
            top1_full_sketch = calculate_accuracy(samples=samples, labels=batch["labels"])
            self.log(f"top1_full_sketch", top1_full_sketch, on_step=False, on_epoch=True, prog_bar=True,
                     logger=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        lr = self.args.lr
        optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        return optimizer
