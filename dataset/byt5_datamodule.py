"""
Dataloader that reads json or json.zip files containing lists of sketches of the form

{"name": "val_00193042", "entities": [[[0, 1], [63, 1]], [[0, 61], [63, 61]]]}

"""

from pathlib import Path

import pytorch_lightning as pl
from adsk_ailab_ray.tools.aws import aws_s3_sync
from datasets import load_dataset
from torch.utils.data import DataLoader

from dataset.dataset_utils import split_list
from dataset.sketch_strings_collator import SketchStringsCollator

from functools import partial

SPECIAL_TOKENS = ["<START_Q>", "<END_Q>", "<START_A>", "<END_A>"]


def collate_fn(batch, tokenizer, max_length):
    input_sequences = [f"<START_Q>{item['question']}<END_Q>"
                       f"<START_A>{item['answer']}<END_A>" 
                       for item in batch]
    out_batch = tokenizer(
        input_sequences,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    out_batch["labels"] = out_batch["input_ids"].clone()

    return out_batch

class Byt5DataModule(pl.LightningDataModule):
    def __init__(self, model_name, batch_size, max_length, min_ratio, max_ratio, s3_data_uri, dataset_path,
                 num_dataloader_workers, tokenizer_cls, extra_val_percentages):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.s3_data_uri = s3_data_uri
        self.dataset_path = dataset_path
        self.num_dataloader_workers = num_dataloader_workers
        self.tokenizer_cls = tokenizer_cls
        self.extra_val_percentages = extra_val_percentages

        self.tokenizer = None
        self.collator = None
        self.ds = None

    def prepare_data(self):
        # Download tokenizer
        self.tokenizer_cls.from_pretrained(self.model_name)
        # Download data
        aws_s3_sync(self.s3_data_uri, self.dataset_path)

    def setup(self, stage):
        self.tokenizer = self.tokenizer_cls.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        SPECIAL_TOKENS = ["<SYSTEM>", "<START_Q>", "<END_Q>", "<START_A>", "<END_A>"]
        self.tokenizer.add_tokens(SPECIAL_TOKENS, special_tokens=True)
        
        self.collator = SketchStringsCollator(tokenizer=self.tokenizer, max_length=self.max_length,
                                              additional_cols=["entities", "input_entities", "output_entities"], model_name=self.model_name)
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

        for p in self.extra_val_percentages:
            cur_ratio_transform = partial(self.batch_split_entities, min_ratio=p/100, max_ratio=p/100)
            ds[f"val_{p}"] = ds["val"].with_transform(cur_ratio_transform)

        return ds

    def train_dataloader(self):
        return self._get_dataloader(ds=self.ds["train"], shuffle=True)

    def val_dataloader(self):
        val_names = self.val_dataloader_names(self.extra_val_percentages)
        return [self._get_dataloader(ds=self.ds[val_name], shuffle=False) for val_name in val_names]

    @staticmethod
    def val_dataloader_names(extra_val_percentages):
        return ["val"] + [f"val_{p}" for p in extra_val_percentages]

    def _get_dataloader(self, ds, shuffle):
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle, collate_fn=self.collator,
                          num_workers=self.num_dataloader_workers, pin_memory=True)

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

        input_text = self.tokenizer.entities_to_str(input_entities, is_prompt=True)
        output_text = self.tokenizer.entities_to_str(output_entities)

        return {
            "input_entities": input_entities,
            "output_entities": output_entities,
            "input_text": input_text,
            "output_text": output_text,
        }
