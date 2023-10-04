"""
Dataloader that reads json or json.zip file containing list sketches with constraints
"""

from pathlib import Path

import pytorch_lightning as pl
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from preprocess.syn_contraints_preprocess import (
    process_for_syn_constraints,
    constraints_to_string,
    pp_constraints_to_string,
    constraints_to_string_schema2,
)

class SketchStringsCollator:
    def __init__(self, tokenizer, max_length=None, additional_cols=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.additional_cols = additional_cols

    def tokenize(self, strings):
        return self.tokenizer(strings, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")

    def __call__(self, examples):
        # Collate input_text and output_text columns
        input_text = [example['input_text'] for example in examples]
        output_text = [example['output_text'] for example in examples]
        name = [example['name'] for example in examples]

        # Encode input and output
        tokenized_input = self.tokenize(input_text)

        tokenized_output = self.tokenize(output_text)
        labels = tokenized_output.input_ids
        # replace padding token id's of the labels by ignore_index=-100 so it's ignored by the loss
        labels[labels == self.tokenizer.pad_token_id] = -100

        batch = {
            "input_ids": tokenized_input.input_ids,
            "attention_mask": tokenized_input.attention_mask,
            "labels": labels,
            "input_text": input_text,
            "output_text": output_text,
            "name": name,
        }

        if self.additional_cols:
            for col in self.additional_cols:
                batch[col] = [example[col] for example in examples]

        return batch

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

    def setup(self, stage, load_from_cache_file=True, num_proc=32):
        self.tokenizer = self.get_tokenizer()
        additional_cols = ["vertices", "edges", "constraints"]
        self.collator = SketchStringsCollator(tokenizer=self.tokenizer, max_length=self.max_length,
                                              additional_cols=additional_cols)
        self.ds = self.get_dataset(load_from_cache_file=load_from_cache_file, num_proc=num_proc)

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

    @staticmethod
    def get_tokenizer(model_name, num_coords=64):
        raise NotImplementedError

    @staticmethod
    def add_input_output_strings(example):
        raise NotImplementedError


class SynConstraintsDataModule(SynConstraintsBaseDataModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def add_input_output_strings(example):
        res = process_for_syn_constraints(example)
        example["entities"] = res["entities"]

        flat_entities = [sum(points, []) for points in example["entities"]]
        entity_strings = ["".join([f"<{x}>" for x in ent]) for ent in flat_entities]

        example["input_text"] = "".join([f"<ent_{i}>{ent}" for i, ent in enumerate(entity_strings)])
        example["output_text"] = constraints_to_string(example["constraints"])

        return example

    @staticmethod
    def get_tokenizer(model_name, num_coords=64):
        num_ent_names = 62
        tokenizer = AutoTokenizer.from_pretrained(model_name)
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
        res = process_for_syn_constraints(example, return_mid_points=True)
        example["entities"] = res["entities"]
        example["mid_points"] = res["mid_points"]

        flat_entities = [sum(points, []) for points in example["entities"]]
        entity_strings = ["".join([f"<{x}>" for x in ent]) + "<ent_sep>" for ent in flat_entities]

        example["input_text"] = "".join(entity_strings)
        example["output_text"] = pp_constraints_to_string(example["constraints"], example["mid_points"])
        return example

    @staticmethod
    def get_tokenizer(model_name, num_coords=64):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        new_tokens = [f"<{i}>" for i in range(num_coords)]
        new_tokens += ["<ent_sep>", "<constraint_sep>", "<parallel_sep>"]
        tokenizer.add_tokens(new_tokens)
        # assert len(tokenizer) % 64 == 0
        return tokenizer


class SynConstraintsSchema2DataModule(SynConstraintsBaseDataModule):
    """
    Distinct constraint tokens
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def add_input_output_strings(example):
        res = process_for_syn_constraints(example)
        example["entities"] = res["entities"]

        flat_entities = [sum(points, []) for points in example["entities"]]
        entity_strings = ["".join([f"<{x}>" for x in ent]) for ent in flat_entities]

        example["input_text"] = "".join([f"<ent_{i}>{ent}" for i, ent in enumerate(entity_strings)])
        example["input_text"] = example["input_text"].replace(";", "")

        example["output_text"] = constraints_to_string_schema2(example["constraints"])
        return example

    @staticmethod
    def get_tokenizer(model_name="google/byt5-small", num_coords=64):
        num_ent_names = 62
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        new_tokens = [f"<{i}>" for i in range(num_coords)] + [f"<ent_{i}>" for i in range(num_ent_names)]
        new_tokens += ["<horizontal>", "<vertical>", "<parallel>", "<perpendicular>", "<parallel_sep>"]
        tokenizer.add_tokens(new_tokens)
        # assert len(tokenizer) % 64 == 0
        return tokenizer