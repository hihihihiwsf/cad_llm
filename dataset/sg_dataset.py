from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import random
import json
from pathlib import Path

from transformers import CLIPImageProcessor
from geometry.parse import get_curves, get_point_entities
from geometry.visualization import visualize_batch, visualize_sample

def add_token(sketch_string):
    ent_list = sketch_string.split(";")[:-1]
    new_ent_list = ''
    for ent in ent_list:
        if len(ent.split(',')) == 4:
            ent = 'L' + ent + ';'
        elif len(ent.split(',')) == 6:
            ent = 'A' + ent + ';'
        elif len(ent.split(',')) == 8:
            ent = 'C' + ent + ';'
        new_ent_list = new_ent_list + ent
    return new_ent_list

class SketchGraphsDataset(Dataset):
    def __init__(self, args, split):
        path = Path(args.dataset) / f"{split}.json" #sg_str_
        with open(path, "r") as f:
            self.data = json.load(f)
        self.args = args
        if split == "train":
            n = int(args.limit_data * len(self.data))
            random.shuffle(self.data)
            self.data = self.data[:n]

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
        if self.args.type_token:
            input_text = add_token(input_text)
            output_text = add_token(output_text)
        sketch_dict['input_text'] = input_text #+'#'
        sketch_dict['output_text'] = output_text #+"#" #suitable for codet5 data features
        return sketch_dict

    def get_mask(self, n):
        """
        Sample a random size for mask and a random mask of size n
        """
        mask_percent = random.uniform(self.min_input_percent, self.max_input_percent)
        mask_size = round(mask_percent * n)
        mask_size = min(max(1, mask_size), n - 1)

        mask = np.zeros(n, dtype=bool)
        mask[:mask_size] = 1
        np.random.shuffle(mask)
        return mask

    def __len__(self):
        return len(self.data)


class SketchGraphsCollator:
    def __init__(self, tokenizer, args, max_length=None):
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.args = args
        self.clip_preprocess = CLIPImageProcessor.from_pretrained(self.args.clipmodel)
        
    def tokenize(self, strings):
        return self.tokenizer(strings, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")

    def __call__(self, sketch_dicts):
        input_strings = [sketch['input_text'] for sketch in sketch_dicts]
        output_strings = [sketch['output_text'] for sketch in sketch_dicts]
        tokenized_input = self.tokenize(input_strings)
        tokenized_output = self.tokenize(output_strings)

        labels = tokenized_output.input_ids
        # replace padding token id's of the labels by ignore_index=-100 so it's ignored by the loss
        # labels[labels == self.tokenizer.pad_token_id] = -100

        point_inputs = [get_point_entities(sketch["input_text"]) for sketch in sketch_dicts]
        input_curves = [get_curves(point_input) for point_input in point_inputs]
        list_of_img = visualize_sample(input_curves=input_curves, box_lim=64 + 3)

        batch_images = self.clip_preprocess(list_of_img, return_tensors="pt")

        batch = {
            "input_ids": tokenized_input.input_ids,
            "attention_mask": tokenized_input.attention_mask,
            "labels": labels,
            "sketches": sketch_dicts,
            "images": batch_images
        }
        return batch


def get_sketchgraphs_dataloader(tokenizer, args, split, shuffle):
    dataset = SketchGraphsDataset(split=split, args=args)
    collator = SketchGraphsCollator(tokenizer=tokenizer, max_length=args.max_length, args=args)
    return DataLoader(dataset, batch_size=args.batch_size, collate_fn=collator, shuffle=shuffle,
                      num_workers=args.num_workers)



def get_fsdp_dataloader(rank, world_size, tokenizer, args):
    train_dataset = SketchGraphsDataset(split='train', args=args)
    val_dataset = SketchGraphsDataset(split='val', args=args)
    collator = SketchGraphsCollator(tokenizer=tokenizer, max_length=args.max_length, args=args)

    train_sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, rank=rank, num_replicas=world_size)


    train_kwargs = {'batch_size': args.batch_size, 'sampler': train_sampler, 'collate_fn':collator}
    test_kwargs = {'batch_size': args.batch_size, 'sampler': val_sampler}

    cuda_kwargs = {'num_workers': 2,
                    'pin_memory': True,
                    'shuffle': False}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_loader = DataLoader(train_dataset,**train_kwargs)
    val_loader = DataLoader(val_dataset, **test_kwargs)
    return train_loader, val_loader