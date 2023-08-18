"""
Dataloader that reads json or json.zip file containing list sketches with constraints
"""

from pathlib import Path

import pytorch_lightning as pl
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from preprocess.syn_contraints_preprocess import get_entities_for_syn_constraints, constraints_to_string, get_pp_constraints_string
from dataset.sketch_strings_collator import SketchStringsCollator


class SynConstraintsBaseDataModule(pl.LightningDataModule):
    def __init__(self, model_name, batch_size, max_length, dataset_path, num_workers):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.dataset_path = dataset_path
        self.num_workers = num_workers

        self.tokenizer = None
        self.collator = None
        self.ds = None

    def prepare_data(self):
        # Download tokenizer
        AutoTokenizer.from_pretrained(self.model_name)

    def setup(self, stage, load_from_cache_file=True):
        self.tokenizer = self.get_tokenizer()
        self.collator = SketchStringsCollator(tokenizer=self.tokenizer, max_length=self.max_length)
        self.ds = self.get_dataset()

    def get_dataset(self, load_from_cache_file=True, num_proc=32):
        # Load dataset
        splits = ["val", "train", "test"]
        data_files = {split: str(Path(self.dataset_path) / f"*{split}.json*") for split in splits}
        ds = load_dataset("json", data_files=data_files, field="data")

        # Process dataset
        ds = ds.rename_columns({"filename": "name"})
        ds = ds.map(self.add_input_output_strings, load_from_cache_file=load_from_cache_file, num_proc=num_proc)
        return ds

    def train_dataloader(self):
        return self._get_dataloader(ds=self.ds["train"], shuffle=True)

    def val_dataloader(self):
        return self._get_dataloader(ds=self.ds["val"], shuffle=False)

    def _get_dataloader(self, ds, shuffle):
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle, collate_fn=self.collator,
                          num_workers=self.num_workers)

    def get_tokenizer(self, num_coords=64):
        raise NotImplementedError

    @staticmethod
    def add_input_output_strings(example):
        raise NotImplementedError


class SynConstraintsDataModule(SynConstraintsBaseDataModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def add_input_output_strings(example):
        entities = get_entities_for_syn_constraints(example)

        example["input_text"] = "".join([f"<ent_{i}>{ent}" for i, ent in enumerate(entities)])
        example["input_text"] = example["input_text"].replace(";", "")

        example["output_text"] = constraints_to_string(example["constraints"])
        return example

    def get_tokenizer(self, num_coords=64):
        num_ent_names = 62
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        new_tokens = [f"<{i}>" for i in range(num_coords)] + [f"<ent_{i}>" for i in range(num_ent_names)]
        new_tokens += ["<constraint_sep>", "<parallel_sep>"]
        tokenizer.add_tokens(new_tokens)
        assert len(tokenizer) % 64 == 0
        return tokenizer


class SynConstraintsPPDataModule(SynConstraintsBaseDataModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def add_input_output_strings(example):
        entities = get_entities_for_syn_constraints(example)
        example["input_text"] = "".join(entities).replace(";", "<ent_sep>")

        example["output_text"] = get_pp_constraints_string(example)
        return example

    def get_tokenizer(self, num_coords=64):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        new_tokens = [f"<{i}>" for i in range(num_coords)]
        new_tokens += ["<ent_sep>", "<constraint_sep>", "<parallel_sep>"]
        tokenizer.add_tokens(new_tokens)
        # assert len(tokenizer) % 64 == 0
        return tokenizer
