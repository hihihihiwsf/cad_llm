from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import json
from pathlib import Path

import torch
import enum


class SketchGraphsDataset(Dataset):
    def __init__(self, args, split):
        path = Path(args.dataset) / f"sg_str_{split}.json"
        with open(path, "r") as f:
            self.data = json.load(f)

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
        sketch_dict['input_text'] = input_text
        sketch_dict['output_text'] = output_text
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

def add_token(sketch_string):
    ent_list = sketch_string.split(";")[:-1]
    new_ent_list = ''
    for ent in ent_list:
        if len(ent.split('>')[0:-1]) == 4:
            ent = '<Line>' + ent + ';'
        elif len(ent.split('>')[0:-1]) == 6:
            ent = '<Curve>' + ent + ';'
        elif len(ent.split('>')[0:-1]) == 8:
            ent = '<Circle>' + ent + ';'
        new_ent_list = new_ent_list + ent
    return new_ent_list

def classify_type(ent):
    if len(ent.split('>')[0:-1]) == 4:
        return 'Line'
    elif len(ent.split('>')[0:-1]) == 6:
        return 'Curve'
    elif len(ent.split('>')[0:-1]) == 8:
        return 'Circle'

def extract(ent):
    v_list = ent.split('<')[1:]
    v_list = [int(v.split('>')[0])+37 for v in v_list] #convert from [-31,32] to[0,63] 
    return v_list

class Token(enum.IntEnum):
    """Enumeration indicating the non-parameter value tokens of PrimitiveModel.
    """
    Pad = 0
    Start = 1
    Stop = 2
    Line = 3
    Curve = 4
    Circle = 5

NUM_PARAMS = {
    'Line': 4,
    'Curve': 6,
    'Circle': 8,
}

NON_COORD_TOKEN = 1 # 0 for padding???
COORD_TOKEN_MAP = {}
tok = NON_COORD_TOKEN + 1
for ent_type in ['Line', 'Curve', 'Circle']:
    COORD_TOKEN_MAP[ent_type] = list(range(tok, tok+NUM_PARAMS[ent_type]))
    tok += NUM_PARAMS[ent_type]

from typing import Callable, Dict, List, Union, Sequence, Tuple, Optional

def _pad_or_truncate_to_length(arr: np.ndarray, target_length: Optional[int]=None):
    if target_length is None:
        return arr
    if len(arr) > target_length:
        return arr[:target_length]
    if isinstance(arr, np.ndarray):
        return np.pad(arr, (0, target_length - len(arr)), constant_values=Token.Pad)
    elif isinstance(arr, torch.Tensor):
        return torch.nn.functional.pad(arr, (0, target_length - len(arr)), value=Token.Pad)
    else:
        raise ValueError('arr must be either numpy array or torch Tensor')

def vitru_tokenize(strings, max_length, padding=True, truncation=True,return_tensors="pt"):
    s_val = np.array([],dtype=int)
    s_coord = np.array([],dtype=int)
    s_pos = np.array([],dtype=int)
    s_mask = np.array([],dtype=int)
    max_len = 0
    for i, sketch in enumerate(strings):

        val, coord, pos, attention_mask = sketch_tokenize(sketch, truncation=True, max_length=max_length,padding=True, return_tensors="pt")
        if max_len==0:
            s_val = val
            s_coord = coord
            s_pos = pos 
            s_mask = attention_mask
        else:
            s_val = np.vstack([s_val, val])
            s_coord = np.vstack([s_coord, coord])
            s_pos = np.vstack([s_pos, pos])
            s_mask = np.vstack([s_mask, attention_mask])
        max_len += 1
    
    sample = {
        'val': s_val,
        'coord': s_coord,
        'pos':s_pos,
        "attention_mask":s_mask
    }
    return sample


def sketch_tokenize(sketch, max_length, padding=True, truncation=True,return_tensors="pt",include_stop=False):
    #val_tokens = [Token.Start]
    val_tokens = []
    coord_tokens = [1] # 0 for padding
    pos_idx = 1  # 0 is reserved for padding
    pos_tokens = [pos_idx]
    
    ent_list = sketch.split(";")[:-1]
    for ent in ent_list:
        val_tokens.append(Token[classify_type(ent)])
        coord_tokens.append(NON_COORD_TOKEN)

        pos_idx += 1
        pos_tokens.append(pos_idx)
        val_tokens.extend(extract(ent))
        coord_tokens.extend(COORD_TOKEN_MAP[classify_type(ent)])
        pos_tokens.extend([pos_idx] * len(extract(ent)))

    if include_stop:
        val_tokens.append(Token.Stop)
        coord_tokens.append(NON_COORD_TOKEN)
        pos_tokens.append(pos_idx+1)
    
    attention_mask =np.ones(np.array(val_tokens).size)
    val = _pad_or_truncate_to_length(np.array(val_tokens, dtype=np.int64), max_length)
    coord= _pad_or_truncate_to_length(np.array(coord_tokens, dtype=np.int64), max_length)
    pos= _pad_or_truncate_to_length(np.array(pos_tokens, dtype=np.int64), max_length)
    attention_mask = _pad_or_truncate_to_length(np.array(attention_mask, dtype=np.int64), max_length)

    return val, coord, pos, attention_mask

class SketchGraphsCollator:
    def __init__(self, tokenizer, max_length=None):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def tokenize(self, strings):

        model_tokenize = self.tokenizer(strings, padding='longest', truncation=True, max_length=self.max_length, return_tensors="pt")

        return model_tokenize 

    def __call__(self, sketch_dicts):

        input_strings_2 = [add_token(sketch['input_text']) for sketch in sketch_dicts]
        output_strings_2 = [add_token(sketch['output_text']) for sketch in sketch_dicts]
        
        tokenized_input = self.tokenize(input_strings_2) #keys: 'input_ids', 'attention_mask'
        tokenized_output = self.tokenize(output_strings_2)

        labels = tokenized_output.input_ids
        # replace padding token id's of the labels by ignore_index=-100 so it's ignored by the loss
        labels[labels == self.tokenizer.pad_token_id] = -100

        batch_1 = {
            "input_ids": tokenized_input.input_ids,
            "attention_mask": tokenized_input.attention_mask,
            "labels": labels,
            "sketches": sketch_dicts
        }

        input_strings = [sketch['input_text'] for sketch in sketch_dicts]
        output_strings = [sketch['output_text'] for sketch in sketch_dicts]

        vitru_tokenized_input = vitru_tokenize(input_strings, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")

        vitru_tokenized_output = vitru_tokenize(output_strings, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        labels = torch.tensor(vitru_tokenized_output['val'])
        labels[labels == Token.Pad] = -100

        batch_input_id = {
            "val_ids": torch.tensor(vitru_tokenized_input['val']),
            "pos_ids": torch.tensor(vitru_tokenized_input['pos']),
            "coord_ids": torch.tensor(vitru_tokenized_input['coord']),
        }

        batch = {
            "input_ids": batch_input_id,
            "attention_mask": torch.tensor(vitru_tokenized_input['attention_mask']),
            "labels": labels,
            "sketches": sketch_dicts
        }

        return batch


def get_sketchgraphs_dataloader(tokenizer, args, split, shuffle, drop_last):
    dataset = SketchGraphsDataset(split=split, args=args)

    # from IPython import embed; embed()
    
    collator = SketchGraphsCollator(tokenizer=tokenizer, max_length=args.max_length)

    return DataLoader(dataset, batch_size=args.batch_size, collate_fn=collator, shuffle=shuffle, drop_last=drop_last,
                      num_workers=args.num_workers)
