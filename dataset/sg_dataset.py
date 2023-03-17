from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import json
from pathlib import Path


class SketchGraphsDataset(Dataset):
    def __init__(self, dataset_dir, split, subset_range=None, ascii_encoding=False):
        path = Path(dataset_dir) / f"sg_str_{split}.json"
        with open(path, "r") as f:
            self.data = json.load(f)

        # Sanity check text format
        entity_string = self.data[0]["entities"][0]
        if not ascii_encoding:
            error_message = f"Expected format '<-31><17>' not '-31, 17', found '{entity_string}'"
            assert entity_string[0] == "<", error_message
            assert "," not in self.data[0]["entities"][0], error_message
        else:
            error_message = f"Expected format '-31, 17' not '<-31><17>', found '{entity_string}'"
            assert entity_string[0] != "<", error_message
            assert "," in self.data[0]["entities"][0], error_message

        self.subset_range = subset_range or [0, 1]
        assert self.subset_range[0] >= 0 and self.subset_range[1] <= 1

    def __getitem__(self, index):
        """
        Applies a random mask to the entities of sketch
        Returns (input_text, output_text)
        """
        sketch_dict = self.data[index]
        entities = sketch_dict["entities"]
        mask = self.get_mask(len(entities))
        sketch_dict["mask"] = mask
        input_text = "".join([ent for i, ent in enumerate(entities) if mask[i]])
        output_text = "".join([ent for i, ent in enumerate(entities) if not mask[i]])
        return input_text, output_text, self.get_sketch(index)

    def get_mask(self, n):
        """
        Sample a random size for mask and a random mask of size n
        """
        mask_size = random.randint(1, n - 1)
        low, high = self.subset_range
        mask_size = min(max(mask_size, int(low * n)), int(high * n))
        mask = np.zeros(n, dtype=bool)
        mask[:mask_size] = 1
        np.random.shuffle(mask)
        return mask

    def get_sketch(self, index):
        """
        Returns the element at index without resampling a mask
        """
        return self.data[index]

    def __len__(self):
        return len(self.data)


class SketchGraphsCollator:
    def __init__(self, tokenizer, max_length=None):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def tokenize(self, strings):
        return self.tokenizer(strings, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")

    def __call__(self, input_output_pairs):
        input_strings = [x for x, _, _ in input_output_pairs]
        output_strings = [y for _, y, _ in input_output_pairs]
        sktch = [z for _, _, z in input_output_pairs]

        tokenized_input = self.tokenize(input_strings)
        tokenized_output = self.tokenize(output_strings)

        labels = tokenized_output.input_ids
        # replace padding token id's of the labels by ignore_index=-100 so it's ignored by the loss
        labels[labels == self.tokenizer.pad_token_id] = -100

        batch = {
            "input_ids": tokenized_input.input_ids,
            "attention_mask": tokenized_input.attention_mask,
            "labels": labels,
            "sketch": sktch
        }
        return batch


def get_sketchgraphs_dataloader(tokenizer, dataset_dir, split, hps, shuffle, num_workers, ascii_encoding):
    dataset = SketchGraphsDataset(dataset_dir=dataset_dir, split=split, subset_range=hps.get("subset_range"),
                                  ascii_encoding=ascii_encoding)
    collator = SketchGraphsCollator(tokenizer=tokenizer, max_length=hps.get("max_length"))
    return DataLoader(dataset, batch_size=hps['batch_size'], collate_fn=collator, shuffle=shuffle,
                      num_workers=num_workers)
