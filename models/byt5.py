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
from transformers.modeling_utils import unwrap_model
import sys

sys.path.insert(0, '/home/ec2-user/SageMaker/efs/code/cad_llm')
from metrics import calculate_accuracy, calculate_first_ent_accuracy, calculate_validity
from util import get_quantized_range
from geometry.parse import parse_string_to_curves
from geometry.visualization import visualize_batch
from pathlib import Path


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

        self.lr = self.args.lr
        self.quantization_bits = 6  # Hard code for now
        self.quantized_range = get_quantized_range(self.quantization_bits)
        self.box_lim = max(self.quantized_range)  # for visualization

        # If using single token encoding - adjust tokenizer and model embeddings
        if not args.ascii_encoding:
            self.adjust_to_use_new_tokens()

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

        # Generate samples and calculate accuracy
        # Recursively unwrap the model from potential distributed training containers
        generate_func = unwrap_model(self.model).generate
        samples = generate_func(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                do_sample=False, max_new_tokens=batch["labels"].shape[1])
        top1_full_sketch = calculate_accuracy(samples=samples, labels=batch["labels"])
        self.log(f"top1_full_sketch", top1_full_sketch, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Decode samples
        string_samples = self.tokenizer.batch_decode(samples, skip_special_tokens=True)
        string_labels = [sketch["output_text"] for sketch in batch["sketches"]]
        top1_ent = calculate_first_ent_accuracy(string_labels=string_labels, string_samples=string_samples)
        self.log(f"top1_ent", top1_ent, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Convert string entities to curves and check validity
        sample_curves = [parse_string_to_curves(string_sample) for string_sample in string_samples]
        validity = calculate_validity(batch_sample_curves=sample_curves)
        self.log(f"validity", validity, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # # Plot sketches
        if batch_idx < 5:
            self.log_samples(batch=batch, sample_curves=sample_curves, batch_idx=batch_idx)

        return loss

    def log_samples(self, batch, sample_curves, batch_idx):
        input_curves = [parse_string_to_curves(sketch["input_text"]) for sketch in batch["sketches"]]
        label_curves = [parse_string_to_curves(sketch["output_text"]) for sketch in batch["sketches"]]

        fig = visualize_batch(input_curves=input_curves, label_curves=label_curves, sample_curves=sample_curves,
                              box_lim=self.box_lim + 3)
        fig_path = Path(self.args.samples_dir) / f"epoch_{self.current_epoch}_batch_{batch_idx}.png"
        fig.savefig(fig_path)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        return optimizer
