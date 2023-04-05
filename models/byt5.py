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
from transformers import T5Config, T5ForConditionalGeneration, AutoTokenizer, T5TokenizerFast
from transformers.modeling_utils import unwrap_model
import sys  
sys.path.insert(0, '/home/ec2-user/SageMaker/efs/code/cad_llm')
from metrics import calculate_accuracy
from util import get_quantized_range
import math

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
        # self.tokenizer = AutoTokenizer.from_pretrained("t5-base-vocab-100")

        self.args = args

        # If using single token encoding - adjust tokenizer and model embeddings
        if not args.ascii_encoding:
            self.adjust_to_use_new_tokens()


    def positionalencoding1d(self, d_model, length):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                            "odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                            -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe

    def adjust_to_use_new_tokens(self):
        # Add new tokens to the tokenizer
        quantize_n_bits = 6  # Hard code for now
        new_tokens = [f"<{i}>" for i in get_quantized_range(quantize_n_bits)]
        self.tokenizer.add_tokens(new_tokens)

        # Add new token embeddings and initialize using learned embeddings
        self.model.resize_token_embeddings(len(self.tokenizer))
        embedding_params = self.model.get_input_embeddings().weight.data
        # pe = self.positionalencoding1d(embedding_params.shape[1], len(new_tokens))


        pe = self.positionalencoding1d(embedding_params.shape[1], int(len(new_tokens)/2)+1)
        b = -1 * torch.flip(pe[1:, :], [0])
        pe = torch.concatenate((b, pe), 0)

        for i, j in enumerate(new_tokens):
            # start with the embedding for 'A', ensures no clash with embedding for ';'
            # embedding_params[-len(new_tokens)+i] = torch.mean(embedding_params[self.tokenizer.encode(j)[:-1]], 0)
            embedding_params[-len(new_tokens)+i] = embedding_params[68] + pe[i, :]
            # embedding_params[-i-1] = embedding_params[65+i]

    def training_step(self, batch, batch_idx):
        # pl.utilities.memory.garbage_collection_cuda()
        cols = ["input_ids", "attention_mask", "labels"]
        model_batch = {col: val for col, val in batch.items() if col in cols}
        outputs = self.model(**model_batch)
        loss = outputs.loss  # CrossEntropyLoss(ignore_index=-100) between outputs.logits and labels
        self.log(f"train_loss", loss.detach().cpu(), on_step=False, on_epoch=True, prog_bar=False, logger=True, batch_size=self.args.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        cols = ["input_ids", "attention_mask", "labels"]
        model_batch = {col: val for col, val in batch.items() if col in cols}
        outputs = self.model(**model_batch)
        loss = outputs.loss
        self.log(f"val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size)

        # Generate samples and calculate accuracy
        # Recursively unwrap the model from potential distributed training containers
        generate_func = unwrap_model(self.model).generate
        samples = generate_func(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                do_sample=False, max_new_tokens=batch["labels"].shape[1])
        top1_full_sketch = calculate_accuracy(samples=samples, labels=batch["labels"])
        self.log(f"top1_full_sketch", top1_full_sketch, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size)

        # return loss

    def configure_optimizers(self):
        lr = self.args.lr
        optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        return optimizer
