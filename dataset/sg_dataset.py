from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import json
from pathlib import Path


class SketchGraphsDataset(Dataset):
    def __init__(self, args, split):
        path = Path(args.dataset) / f"sg_str_{split}.json"
        with open(path, "r") as f:
            self.data = json.load(f)

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
        mask = self.get_mask(len(entities))
        sketch_dict["mask"] = mask
        input_text = "".join([ent for i, ent in enumerate(entities) if mask[i]])
        output_text = "".join([ent for i, ent in enumerate(entities) if not mask[i]])
        sketch_dict['input_text'] = input_text
        sketch_dict['output_text'] = output_text
        return sketch_dict

    def get_mask(self, n):
        """
        Sample a random size for mask and a random mask of size n
        """
        mask_size = random.randint(1, n - 1)
        low, high = self.min_input_percent, self.max_input_percent
        mask_size = min(max(mask_size, int(low * n)), int(high * n))
        mask = np.zeros(n, dtype=bool)
        mask[:mask_size] = 1
        np.random.shuffle(mask)
        return mask

    def __len__(self):
        return len(self.data)


class SketchGraphsCollator:
    def __init__(self, tokenizer, max_length=None):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def tokenize(self, strings):
        return self.tokenizer(strings, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")

    def __call__(self, sketch_dicts):
        
        sep_token = '[SEP]'
        input_strings = [self.tokenizer.bos_token + sketch['input_text'] + sep_token for sketch in sketch_dicts]
        output_strings = [sketch['output_text'] for sketch in sketch_dicts]

        new_value = self.tokenizer.mask_token
        all_strings = [self.tokenizer.bos_token + sketch['input_text'] + sep_token + sketch['output_text'] + self.tokenizer.eos_token for sketch in sketch_dicts]
        # input_strings = [sketch['input_text'] + sep_token + new_value* len(self.tokenizer.encode(sketch['output_text'])) + self.tokenizer.eos_token for sketch in sketch_dicts]
        # all_strings_nocorruption = ["".join(sketch['entities']) for sketch in sketch_dicts]
        # all_strings = []
        # for i, s in enumerate(sketch_dicts):
        #     
        #     lst = [new_value * len(self.tokenizer.encode(val)) if m == False else val for m, val in zip(s['mask'], s['entities'])]
        #     # all_strings.append("".join(lst) + self.tokenizer.eos_token + s['output_text'])
        #     all_strings.append("".join(lst))


        tokenized_input = self.tokenize(input_strings)
        tokenized_output = self.tokenize(output_strings)
        tokenized_all = self.tokenize(all_strings)
        labels_all = self.tokenize(all_strings)

        # labels_all = tokenized_all.input_ids.clone()
        # replace padding token id's of the labels by ignore_index=-100 so it's ignored by the loss
        labels_all[labels_all == self.tokenizer.pad_token_id] = -100

        labels = tokenized_output.input_ids
        labels[labels == self.tokenizer.pad_token_id] = -100
        

        batch = {
            "input_ids": tokenized_all.input_ids,
            "attention_mask": tokenized_all.attention_mask,
            "labels": tokenized_all.input_ids,

            "labels_out": labels,
            "input_ids_input": tokenized_input.input_ids,
            "attention_mask_input": tokenized_input.attention_mask,

            "sketches": sketch_dicts
        }
        return batch


def get_sketchgraphs_dataloader(tokenizer, args, split, shuffle):
    dataset = SketchGraphsDataset(split=split, args=args)
    collator = SketchGraphsCollator(tokenizer=tokenizer, max_length=args.max_length)
    return DataLoader(dataset, batch_size=args.batch_size, collate_fn=collator, shuffle=shuffle,
                      num_workers=args.num_workers)
