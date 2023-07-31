import subprocess
import time
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

from dataset.dataset_utils import split_list
from dataset.sketch_strings_collator import SketchStringsCollator


class SketchGraphsDataset(Dataset):
    def __init__(self, args, split):
        # Load dataset from json or json.zip file
        data_files = {split: str(Path(args.dataset) / f"{split}.json*")}
        self.data = load_dataset("json", data_files=data_files)[split]

        if split == "train" and args.limit_data != 1:
            n = int(args.limit_data * len(self.data))
            self.data = self.data.shuffle(seed=args.seed)
            self.data = self.data.select(range(n))

        self.order = args.train_order if split == "train" else "sorted"
        assert self.order in ["sorted", "user", "random"]
        self.entities_col = "user_ordered_entities" if self.order == "user" else "entities"

        # Sanity check text format
        entity_string = self.data[0][self.entities_col][0]
        if args.ascii_encoding:
            error_message = f"Expected format '-31, 17' not '<-31><17>', found '{entity_string}'"
            assert entity_string[0] != "<", error_message
            assert "," in self.data[0][self.entities_col][0], error_message
        else:
            error_message = f"Expected format '<-31><17>' not '-31, 17', found '{entity_string}'"
            assert entity_string[0] == "<", error_message
            assert "," not in self.data[0][self.entities_col][0], error_message

        self.min_input_percent = args.min_input_percent
        self.max_input_percent = args.max_input_percent
        assert self.min_input_percent >= 0 and self.max_input_percent <= 1

    def __getitem__(self, index):
        """
        Applies a random mask to the entities of sketch
        Returns (input_text, output_text)
        """
        sketch_dict = self.data[index]
        entities = sketch_dict[self.entities_col]
        if self.order == "random":
            np.random.shuffle(entities)

        input_entities, output_entities = split_list(entities, self.min_input_percent, self.max_input_percent)
        sketch_dict['input_text'] = "".join(input_entities)
        sketch_dict['output_text'] = "".join(output_entities)

        return sketch_dict

    def __len__(self):
        return len(self.data)



def get_sketchgraphs_dataloader(tokenizer, args, split, shuffle):
    dataset = SketchGraphsDataset(split=split, args=args)
    collator = SketchStringsCollator(tokenizer=tokenizer, max_length=args.max_length)
    return DataLoader(dataset, batch_size=args.batch_size, collate_fn=collator, shuffle=shuffle,
                      num_workers=args.num_workers)


class SketchGraphsDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, args, ray_args):
        super().__init__()
        self.tokenizer = tokenizer
        self.args = args
        self.ray_args = ray_args

    def setup(self, stage):
        self.aws_s3_sync(f"s3://{self.ray_args.input_s3_bucket}", self.args.dataset)

    def train_dataloader(self):
        return get_sketchgraphs_dataloader(
                tokenizer=self.tokenizer,
                args=self.args,
                split="train",
                shuffle=True
        )

    def val_dataloader(self):
        return get_sketchgraphs_dataloader(
                tokenizer=self.tokenizer,
                args=self.args,
                split="val",
                shuffle=False
        )
    
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
