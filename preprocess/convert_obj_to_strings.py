import numpy as np
import json
from preprocess_utils import get_files, get_output_dir, save_splits
from preprocessing import preprocess_sketch
import argparse
from deduplicate import deduplicate_splits
from tqdm import tqdm


def convert_obj_data_to_strings(obj_data, offset):
    string_data = []

    chunk_size = 50000
    for sketch in obj_data:
        sketch["name"] = offset * chunk_size + sketch["index"]
        string_sketch = preprocess_sketch(sketch, quantize_bits=6, new_tokens=True)
        if not string_sketch:
            continue
        string_data.append(string_sketch)

    return string_data


def get_split_and_offset(input_file_path):
    filename = input_file_path.stem
    if "_" in filename:
        split, offset_str = filename.split("_")
    else:
        split, offset_str = filename, 0

    if split == "valid":
        split = "val"

    return split, int(offset_str)

def get_index_list(sketches):
    return [sketch["name"] for sketch in sketches]

def main(args):
    input_file_paths = get_files(args.input, pattern="*.npy")
    input_file_paths.sort()

    output_dir = get_output_dir(args.output)

    splits = ["test", "train", "val"]
    split_to_sketches = {split: [] for split in splits}

    print("Loading and converting data")
    for input_file_path in tqdm(input_file_paths):
        split, offset = get_split_and_offset(input_file_path)

        obj_data = np.load(input_file_path, allow_pickle=True)
        string_data = convert_obj_data_to_strings(obj_data, offset=offset)

        split_to_sketches[split].extend(string_data)

    print("Deduplicating data")
    split_to_sketches = deduplicate_splits(split_to_sketches)

    save_splits(output_dir, split_to_sketches)

    split_to_indices = {split: get_index_list(sketches) for split, sketches in split_to_sketches.items()}
    with open(output_dir / "split_to_index.json", "w") as f:
        json.dump(split_to_indices, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input folder containing obj-like json files")
    parser.add_argument("--output", type=str, help="Output folder to save the string data [default: output]")
    args = parser.parse_args()

    main(args)
