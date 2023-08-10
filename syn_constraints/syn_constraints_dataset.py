"""
Dataloader that reads json or json.zip file containing list sketches with constraints
"""

import subprocess
import time
from pathlib import Path

import pytorch_lightning as pl
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from syn_constraints.syn_contraints_preprocess import get_entities_for_syn_constraints, constraints_to_string
from syn_constraints.syn_constraints_collator import SynConstraintsCollator


class SynConstraintsDataModule(pl.LightningDataModule):
    def __init__(self, args, ray_args):
        super().__init__()
        self.args = args
        self.ray_args = ray_args
        self.tokenizer = None
        self.collator = None
        self.ds = None

    def prepare_data(self):
        # Download dataset
        self.aws_s3_sync(f"s3://{self.ray_args.input_s3_bucket}", self.args.dataset)

        # Download tokenizer
        AutoTokenizer.from_pretrained(self.args.model_name)

    @staticmethod
    def aws_s3_sync(source, destination):
        cmd = ["aws", "s3", "sync", "--quiet", source, destination]
        print(f"Syncing files from {source} to {destination}")
        start_time = time.time()
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p.wait()
        end_time = time.time()
        print("Time Taken to Sync: ", (end_time - start_time))
        return

    def setup(self, stage, load_from_cache_file=True):
        self.tokenizer = self.get_tokenizer()
        self.collator = SynConstraintsCollator(tokenizer=self.tokenizer, max_length=self.args.max_length)

        splits = ["val", "train", "test"]
        data_files = {split: str(Path(self.args.dataset) / f"*{split}.json*") for split in splits}
        ds = load_dataset("json", data_files=data_files, field="data")

        ds = ds.rename_columns({"filename": "name"})
        ds = ds.map(self.add_entities, load_from_cache_file=load_from_cache_file)
        ds = ds.map(self.add_input_string, load_from_cache_file=load_from_cache_file)
        ds = ds.map(self.add_output_string, load_from_cache_file=load_from_cache_file)
        columns_to_remove = ["name", "vertices", "edges", "entities", "constraints", "constraints_seq"]
        ds = ds.remove_columns(column_names=columns_to_remove)
        self.ds = ds

    @staticmethod
    def add_entities(example):
        example["entities"] = get_entities_for_syn_constraints(example, quantize_bits=6, new_tokens=True)
        return example

    @staticmethod
    def add_input_string(example):
        example["input_text"] = "".join([f"<ent_{i}>{ent}" for i, ent in enumerate(example["entities"])])
        example["input_text"] = example["input_text"].replace(";", "")
        return example

    @staticmethod
    def add_output_string(example):
        example["output_text"] = constraints_to_string(example["constraints"])
        return example

    def get_tokenizer(self, num_coords=64, num_ent_names=62):
        tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)
        new_tokens = [f"<{i}>" for i in range(num_coords)] + [f"<ent_{i}>" for i in range(num_ent_names)]
        new_tokens += ["<constraint_sep>", "<parallel_sep>"]
        tokenizer.add_tokens(new_tokens)
        assert len(tokenizer) % 64 == 0
        return tokenizer

    def train_dataloader(self):
        ds = self.ds["train"]
        return DataLoader(ds, batch_size=self.args.batch_size, shuffle=True, collate_fn=self.collator, num_workers=self.args.num_workers)

    def val_dataloader(self):
        ds = self.ds["val"]
        return DataLoader(ds, batch_size=self.args.batch_size, shuffle=False, collate_fn=self.collator, num_workers=self.args.num_workers)
