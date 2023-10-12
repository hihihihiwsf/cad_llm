from filelock import FileLock
import pytorch_lightning as pl
import torch
from pathlib import Path
import os

from transformers.utils.hub import TRANSFORMERS_CACHE
from adsk_ailab_ray.tools.aws import aws_s3_sync, aws_s3_cp
import deepspeed

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    get_linear_schedule_with_warmup,
)


OPTIM_BETAS = (0.9, 0.999)
OPTIM_EPS = 1e-8
OPTIM_WEIGHT_DECAY = 0.0


def get_model_bucket_uri(model_id):
    return f"s3://llama-2-weights/models--{model_id.replace('/', '--')}"


def get_model_download_dir(model_id):
    return TRANSFORMERS_CACHE + f"/models--{model_id.replace('/', '--')}"


def get_model_checkpoint_and_refs_dir(model_id):
    model_bucket_uri = get_model_bucket_uri(model_id)
    model_download_dir = get_model_download_dir(model_id)

    model_hash_path = model_download_dir + "/refs/main"

    if not Path(model_hash_path).exists():
        # Path(model_download_dir + "/refs").mkdir(parents=True, exist_ok=True)
        os.makedirs(model_download_dir + "/refs", exist_ok=True)
        remote_model_hash_uri = model_bucket_uri + "/refs/main"
        aws_s3_cp(remote_model_hash_uri, model_hash_path)

    with open(model_hash_path, "r") as f:
        f_hash = f.read().strip()

    model_ref_path = model_download_dir + "/refs"
    model_checkpoint_path = model_download_dir + "/snapshots/" + f_hash

    return model_checkpoint_path, model_ref_path


def download_model_weights(model_id, model_bucket_uri, model_download_dir):

    base_path = Path(model_download_dir).parent
    base_path.mkdir(parents=True, exist_ok=True)
    lock_file = str(f'{model_id.replace("/",  "--")}.lock')
    with FileLock(lock_file):
        aws_s3_sync(model_bucket_uri, model_download_dir)


class Llama2Model(pl.LightningModule):
    def __init__(self, model_name, model_bucket_uri, model_download_dir, 
                 model_checkpoint_path, batch_size, vocab_size, no_grad_ckpt=False, 
                 num_training_steps=1000, lr=5e-6, strategy='deepspeed', max_length=192,
                 local_samples_path=None, remote_samples_path=None, val_names=None, tokenizer=None):
        super().__init__()

        self.model_id = model_name
        self.model_bucket_uri = model_bucket_uri
        self.model_download_dir = model_download_dir
        self.model_checkpoint_path = model_checkpoint_path
        self.vocab_size = vocab_size
        self.no_grad_ckpt = no_grad_ckpt
        self.num_training_steps = num_training_steps
        self.lr = lr
        self.strategy = strategy
        self.max_length = max_length
        self.batch_size = batch_size

        download_model_weights(self.model_id, self.model_bucket_uri, self.model_download_dir)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_checkpoint_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            # `use_cache=True` is incompatible with gradient checkpointing.
            use_cache=False,
        )

        self.model.resize_token_embeddings(self.vocab_size)

        if not self.no_grad_ckpt:
            self.model.gradient_checkpointing_enable()

    def training_step(self, batch):
        outputs = self.model(**self._get_model_batch(batch))
        loss = outputs.loss

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=self.batch_size)

        return loss

    def validation_step(self, val_batch, batch_idx):
        outputs = self.model(**self._get_model_batch(val_batch))
        loss = outputs.loss
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True,
                 batch_size=self.batch_size)

    def configure_optimizers(self):
            
        if self.strategy == 'deepspeed':
            optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam(self.parameters(), lr=self.lr, betas=OPTIM_BETAS, weight_decay=OPTIM_WEIGHT_DECAY, eps=OPTIM_EPS)
        else:
            optimizer = torch.optim.AdamW(self.trainer.model.parameters(), lr=self.lr, betas=OPTIM_BETAS, weight_decay=OPTIM_WEIGHT_DECAY, eps=OPTIM_EPS)

        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=100,
            num_training_steps=self.num_training_steps,
        )

        return [optimizer], [lr_scheduler]
    
    @staticmethod
    def get_config(model_checkpoint_path):
        # Context for legacy=True: https://github.com/huggingface/transformers/issues/25176
        config = AutoConfig.from_pretrained(model_checkpoint_path, legacy=False)
        
        return config
    
    @staticmethod
    def get_tokenizer(model_checkpoint_path):
        # Context for legacy=True: https://github.com/huggingface/transformers/issues/25176
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint_path, legacy=False)
        tokenizer.pad_token = tokenizer.eos_token
        # tokenizer.add_tokens(SPECIAL_TOKENS, special_tokens=True)

        return tokenizer

    def _get_model_batch(self, batch):
        cols = ["input_ids", "attention_mask", "labels"]
        return {col: val for col, val in batch.items() if col in cols}