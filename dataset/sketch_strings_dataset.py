"""
Dataset that reads json or json.zip file containing list sketches.
Each sketch is a dict containing a list entities
[{"entities": [[[x1, y1], [x2, y3]], ....]} ...]
"""

from functools import partial
from pathlib import Path

from datasets import load_dataset

from dataset.dataset_utils import split_list


def get_sketch_strings_dataset(path, min_split_ratio=0.2, max_split_ratio=0.8):
    splits = ["val", "train", "test"]
    data_files = {split: str(Path(path) / f"{split}.json*") for split in splits}

    dataset = load_dataset("json", data_files=data_files)

    # Add transform to split to input/output text
    # Note that a new random split is generated on each call
    _transform = partial(batch_split_entities_to_io, min_ratio=min_split_ratio, max_ratio=max_split_ratio)
    dataset.set_transform(_transform)
    return dataset


def batch_split_entities_to_io(batch, min_ratio, max_ratio):
    io_triplets = [split_entities_to_io(entities, min_ratio, max_ratio) for entities in batch["entities"]]
    return {
        "input_text": [input_text for input_text, output_text, length_list in io_triplets], #[input_text for input_text, output_text in io_pairs],
        "output_text": [output_text for input_text, output_text, length_list in io_triplets], #[output_text for input_text, output_text in io_pairs],
        "length": [length_list for input_text, output_text, length_list in io_triplets],
    }


def split_entities_to_io(entities, min_ratio, max_ratio):
    # Split
    input_entities, output_entities = split_list(entities, min_ratio, max_ratio)
    # Convert to strings
    input_text = get_entities_string(input_entities)
    output_text = get_entities_string(output_entities)
    length_list = [len(i_t)+len(o_t) for i_t,o_t in zip(input_text, output_text)]
    return input_text, output_text, length_list


def get_entities_string(entities):
    entity_string_list = [get_entity_string(entity) for entity in entities]
    return "".join(entity_string_list)


def get_entity_string(entity):
    point_strings = [f"<{x}><{y}>" for x, y in entity]
    return "".join(point_strings) + ";"
