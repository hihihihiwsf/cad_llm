"""
ByT5 pytorch lightning model
"""

try:
    import comet_ml  # Import before torch
except ImportError:
    pass
import torch
import torch.optim as optim

# import pytorch_lightning as pl
import lightning.pytorch as pl

from transformers import GPTNeoForCausalLM, GPT2Tokenizer, AutoTokenizer, EncoderDecoderModel, GPTNeoModel, GPT2LMHeadModel
from transformers.modeling_utils import unwrap_model
import sys
import math
import math

sys.path.insert(0, '/home/ec2-user/SageMaker/efs/code/cad_llm')
from metrics import calculate_accuracy, calculate_first_ent_accuracy, calculate_validity
from util import get_quantized_range
from geometry.parse import get_curves, get_point_entities
from geometry.visualization import visualize_batch
from pathlib import Path



# model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
# tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

# prompt = (
#     "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
#     "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
#     "researchers was the fact that the unicorns spoke perfect English."
# )

# input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# gen_tokens = model.generate(
#     input_ids,
#     do_sample=True,
#     temperature=0.9,
#     max_length=100,
# )
# gen_text = tokenizer.batch_decode(gen_tokens)[0]

class GPT_Neo(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        if args.untrained_model:
            # config = T5Config.from_pretrained(args.model_name)
            # model = T5ForConditionalGeneration(config)
            model = GPTNeoForCausalLM.from_pretrained(args.model_name)
            model._init_weights(model)  # maybe redundant
        else:
            model = GPTNeoForCausalLM.from_pretrained(args.model_name)
            # model = GPT2LMHeadModel.from_pretrained(args.model_name)
        

        # self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(args.model_name, args.model_name)
        # self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        # self.tokenizer = GPT2Tokenizer.from_pretrained(args.model_name, padding_side='left')
        # self.tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side='left', sep_token="<delim>")
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name, sep_token='[SEP]', pad_token='<pad>', padding_side='left')

        self.model = model
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # self.tokenizer.pad_token = self.tokenizer.bos_token
        if self.tokenizer.sep_token is None:
            self.tokenizer.add_special_tokens({'sep_token': '[SEP]'})
        
        self.model.resize_token_embeddings(len(self.tokenizer))


        # self.model.config.decoder_start_token_id = self.tokenizer.bos_token_id
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.args = args

        self.lr = self.args.lr
        self.batch_size = self.args.batch_size  # to fix logging warning
        self.quantization_bits = 6  # Hard code for now
        self.quantized_range = get_quantized_range(self.quantization_bits)
        self.box_lim = max(self.quantized_range)  # for visualization

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
        new_tokens = [f"<{i}>" for i in self.quantized_range]
        # new_tokens.append(delimiter_token)
        self.tokenizer.add_tokens(new_tokens)
        # self.tokenizer.sep_token = delimiter_token


        # Add new token embeddings and initialize using learned embeddings
        self.model.resize_token_embeddings(len(self.tokenizer))
        embedding_params = self.model.get_input_embeddings().weight.data
        pe = self.positionalencoding1d(embedding_params.shape[1], len(new_tokens))


        # pe = self.positionalencoding1d(embedding_params.shape[1], int(len(new_tokens)/2)+1)
        # b = -1 * torch.flip(pe[1:, :], [0])
        # pe = torch.concatenate((b, pe), 0)



        # embedding_params = self.model.get_input_embeddings().weight.data
        for i in range(1, len(new_tokens)+1):
            # start with the embedding for 'A', ensures no clash with embedding for ';'
            embedding_params[-i] = embedding_params[31] + pe[i-1, :] #"A" starts from 31 for gpt2 tokenizer

    def training_step(self, batch, batch_idx):

        cols = ["input_ids", "attention_mask", "labels"]
        model_batch = {col: val for col, val in batch.items() if col in cols}
        outputs = self.model(**model_batch)
        loss = outputs.loss  # CrossEntropyLoss(ignore_index=-100) between outputs.logits and labels
        self.log(f"train_loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True,
                 batch_size=self.batch_size)
        self.log(f"loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True,
                batch_size=self.batch_size)
        # self.log(f"lr", self.lr, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):

        cols = ["input_ids", "attention_mask", "labels"]
        model_batch = {col: val for col, val in batch.items() if col in cols}
        outputs = self.model(**model_batch)
        loss = outputs.loss
        self.log(f"val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=self.batch_size)

        batch["input_ids"] = batch["input_ids_input"]
        batch["attention_mask"] = batch["attention_mask_input"]
        batch["labels"] = batch["labels_out"]
        # Generate and process samples
        self.generate_samples(batch)

        # Calculate metrics
        top1_full_sketch = calculate_accuracy(samples=batch["point_samples"], labels=batch["point_labels"])
        self.log(f"top1_full_sketch", top1_full_sketch, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=self.batch_size)

        top1_ent = calculate_first_ent_accuracy(samples=batch["point_samples"], labels=batch["point_labels"])
        self.log(f"top1_ent", top1_ent, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=self.batch_size)

        # Convert string entities to curves and check validity
        validity = calculate_validity(batch_sample_curves=batch["sample_curves"])
        self.log(f"validity", validity, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=self.batch_size)

        # # Plot sketches
        if batch_idx < 5:
            self.log_samples(batch=batch, batch_idx=batch_idx)

        return loss

    def generate_samples(self, batch):
        # Recursively unwrap the model from potential distributed training containers
        generate_func = unwrap_model(self.model).generate
        batch["samples"] = generate_func(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                         do_sample=False, max_new_tokens=batch["labels"].shape[1], pad_token_id=self.tokenizer.pad_token_id)


        first_special_token_occurance = []
        for s in batch["samples"]:
            flag = 0
            for i, x in enumerate(s):
                # if x in self.tokenizer.all_special_ids:
                if x == self.tokenizer.sep_token_id:
                    first_special_token_occurance.append(i+1)
                    flag = 1
                    break
            if not flag:
                first_special_token_occurance.append(0)
        

        # batch["string_samples"] = self.tokenizer.batch_decode(batch["samples"], skip_special_tokens=True)
        batch["string_samples"] = []
        for i, s in enumerate(batch['samples']):
            batch['string_samples'].append(self.tokenizer.decode(s[first_special_token_occurance[i]:], skip_special_tokens=True))

        batch["string_labels"] = [sketch["output_text"] for sketch in batch["sketches"]]

        # Cutting the outputs from the first special token occurance
        batch["point_samples"] = [get_point_entities(string_sample) for i, string_sample in enumerate(batch["string_samples"])]
        batch["point_labels"] = [get_point_entities(string_label) for string_label in batch["string_labels"]]

        batch["sample_curves"] = [get_curves(point_sample) for point_sample in batch["point_samples"]]

    def log_samples(self, batch, batch_idx):
        label_curves = [get_curves(point_label) for point_label in batch["point_labels"]]

        point_inputs = [get_point_entities(sketch["input_text"]) for sketch in batch["sketches"]]
        input_curves = [get_curves(point_input) for point_input in point_inputs]

        fig = visualize_batch(input_curves=input_curves, label_curves=label_curves,
                              sample_curves=batch["sample_curves"], box_lim=self.box_lim + 3)
        fig_path = Path(self.args.samples_dir) / f"epoch_{self.current_epoch}_batch_{batch_idx}.png"
        fig.savefig(fig_path)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        # optimizer = DeepSpeedCPUAdam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=200)
        return optimizer
