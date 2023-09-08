"""
Dataloader that reads json or json.zip files containing lists of sketches of the form

{"name": "val_00193042", "entities": [[[0, 1], [63, 1]], [[0, 61], [63, 61]]]}

"""

from pathlib import Path

import pytorch_lightning as pl
from adsk_ailab_ray.tools.aws import aws_s3_sync
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset.dataset_utils import split_list
from dataset.sketch_strings_collator import SketchStringsCollator

from functools import partial


class Byt5NewTokensDataModule(pl.LightningDataModule):

    END_OF_ENT_TOKEN_STR = "<EOE>"
    END_OF_PROMPT_TOKEN_STR = "<EOP>"

    def __init__(self, model_name, batch_size, max_length, min_ratio, max_ratio, input_s3_bucket, dataset_path,
                 num_dataloader_workers):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.input_s3_bucket = input_s3_bucket
        self.dataset_path = dataset_path
        self.num_dataloader_workers = num_dataloader_workers

        self.tokenizer = None
        self.collator = None
        self.ds = None

    def prepare_data(self):
        # Download tokenizer
        AutoTokenizer.from_pretrained(self.model_name)
        # Download data
        aws_s3_sync(f"s3://{self.input_s3_bucket}", self.dataset_path)

    def setup(self, stage):
        self.tokenizer = self.get_tokenizer(self.model_name)
        self.collator = SketchStringsCollator(tokenizer=self.tokenizer, max_length=self.max_length,
                                              additional_cols=["entities", "input_entities", "output_entities"])
        self.ds = self.get_dataset()

    def get_dataset(self):
        # Load dataset
        splits = ["val", "train", "test"]
        data_files = {split: str(Path(self.dataset_path) / f"*{split}.json*") for split in splits}
        ds = load_dataset("json", data_files=data_files)

        # Process dataset
        # Add transform to split to input/output text
        # Note that a new random split is generated on each call
        transform = partial(self.batch_split_entities, min_ratio=self.min_ratio, max_ratio=self.max_ratio)
        ds["train"] = ds["train"].with_transform(transform)
        ds["val"] = ds["val"].with_transform(transform)

        for p in [20, 40, 60, 80]:
            cur_ratio_transform = partial(self.batch_split_entities, min_ratio=p/100, max_ratio=p/100)
            ds[f"val_{p}"] = ds["val"].with_transform(cur_ratio_transform)

        return ds

    def train_dataloader(self):
        return self._get_dataloader(ds=self.ds["train"], shuffle=True)

    def val_dataloader(self):
        return [
            self._get_dataloader(ds=self.ds["val"], shuffle=False),
            self._get_dataloader(ds=self.ds["val_20"], shuffle=False),
            self._get_dataloader(ds=self.ds["val_40"], shuffle=False),
            self._get_dataloader(ds=self.ds["val_60"], shuffle=False),
            self._get_dataloader(ds=self.ds["val_80"], shuffle=False),
        ]

    def _get_dataloader(self, ds, shuffle):
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle, collate_fn=self.collator,
                          num_workers=self.num_dataloader_workers)

    def batch_split_entities(self, batch, min_ratio, max_ratio):
        """ Wrapper for split_entities_to_io """
        results = [self.split_entities_to_io(entities, min_ratio, max_ratio) for entities in batch["entities"]]

        batch["input_entities"] = [result["input_entities"] for result in results]
        batch["output_entities"] = [result["output_entities"] for result in results]

        batch["input_text"] = [result["input_text"] for result in results]
        batch["output_text"] = [result["output_text"] for result in results]

        return batch

    def split_entities_to_io(self, entities, min_ratio, max_ratio):
        # Split
        input_entities, output_entities = split_list(entities, min_ratio, max_ratio)
        # Convert to strings
        input_text = self.get_entities_string(input_entities) + self.END_OF_PROMPT_TOKEN_STR
        output_text = self.get_entities_string(output_entities)

        return {
            "input_entities": input_entities,
            "output_entities": output_entities,
            "input_text": input_text,
            "output_text": output_text,
        }

    def get_entities_string(self, entities):
        entity_string_list = [self.get_entity_string(entity) for entity in entities]
        return self.END_OF_ENT_TOKEN_STR.join(entity_string_list)

    def get_entity_string(self, entity):
        assert all(0 <= coord <= 63 for point in entity for coord in point)
        return "".join([f"<{x}><{y}>" for x, y in entity])

    @staticmethod
    def get_tokenizer(model_name):
        num_coords = 64
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        new_tokens = [f"<{i}>" for i in range(num_coords)]
        new_tokens += [Byt5NewTokensDataModule.END_OF_ENT_TOKEN_STR, Byt5NewTokensDataModule.END_OF_PROMPT_TOKEN_STR]
        tokenizer.add_tokens(new_tokens)
        return tokenizer
