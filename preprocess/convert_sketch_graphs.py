"""

Convert the SketchGraphs Dataset sketches
into a json file containing lists of entities

Versions:
sg_strings_v2: deduped, single token strings
sg_strings_v3: deduped, single token strings, user order option
sg_strings_v4: filtered for SolidGen, deduped, single token strings, user order option

"""
# # hack for running locally
# import os
#
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import time
import argparse
import importlib
import json
from tqdm import tqdm
from sketchgraphs.data import flat_array, sketch_from_sequence, sketch

import sketch_sg
from preprocessing import preprocess_sketch
from deduplicate import deduplicate_splits

importlib.reload(sketch_sg)
from sketch_sg import SketchSG




def get_files(args):
    """Get the Sketch Graphs files to process"""
    input_dir = Path(args.input)
    if not input_dir.exists():
        print("Input folder does not exist")
        exit()
    if not input_dir.is_dir():
        print("Input folder is not a directory")
        exit()
    files = [f for f in input_dir.glob("*.npy")]
    # files = [ input_dir / "sg_t16_validation.npy" ]
    if len(files) == 0:
        print("No SketchGraphs files found")
        exit()
    return files


def get_output_dir(args):
    """Get the output directory to save the data"""
    current_dir = Path(__file__).resolve().parent
    if args.output is not None:
        output_dir = Path(args.output)
    else:
        output_dir = current_dir / "output"
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    return output_dir


def load_filter(path):
    """
        Expected filter file format:
        {
            'train': ['train_07265623.obj', ...],
            'val': ['val_00032698.obj', ...],
            'test': ['test_00289446.obj', ...]
        }
    """
    with open(path) as f:
        split_to_filenames = json.load(f)
    return split_to_filenames


def get_index(filename):
    """Example filename: train_07265623"""
    index_str = filename.split('.')[0].split('_')[1]
    return int(index_str)


def get_split_name(path):
    """Get split name from path e.g. '.../sg_t16_validation.npy' -> 'val' """
    split_name = path.stem.split("_")[2]
    if split_name == "validation":
        split_name = "val"
    return split_name


def convert_sketch(index, seq, split_name, quantize_bits, new_tokens):
    """Convert a sketch from the SketchGraphs dataset"""
    sketch = sketch_from_sequence(seq)
    sketch_name = f"{split_name}_{index:08}"
    sketch_obj = SketchSG(sketch, sketch_name)
    
    '''add constraints information'''
    sketch_dict = sketch_obj.convert_with_constraints()

    return preprocess_sketch(sketch_dict=sketch_dict, quantize_bits=quantize_bits, new_tokens=new_tokens)


def convert_split(file, split_name, filter_filenames, limit, quantize_bits, new_tokens):
    """Convert the sketches of a given SketchGraphs dataset split"""
    seq_data = flat_array.load_dictionary_flat(file)
    seq_count = len(seq_data["sequences"])
    seq_limit = seq_count

    if limit is not None:
        if split_name == "test" or split_name == "val":
            seq_limit = int(limit * 0.2)
        elif split_name == "train":
            seq_limit = int(limit * 0.6)

    print(f"Converting {seq_limit} {split_name} designs...")
    sketch_str_dicts = []
    for filename in tqdm(filter_filenames[:seq_limit]):
        index = get_index(filename)
        seq = seq_data["sequences"][index]
        sketch_dict = convert_sketch(index=index, seq=seq, split_name=split_name, quantize_bits=quantize_bits,
                                     new_tokens=new_tokens)
        if not sketch_dict:
            continue
        sketch_str_dicts.append(sketch_dict)

    return sketch_str_dicts

def save_splits(output_dir, split_to_sketches):
    for split_name, sketches in split_to_sketches.items():
        filename = output_dir / f"{split_name}.json"
        with open(filename, "w") as f:
            json.dump(sketches, f)


def filter_main(sg_files, output_dir, filter_path, limit, quantize_bits, new_tokens):
    split_to_filenames = load_filter(filter_path)
    assert split_to_filenames.keys() == {"train", "val", "test"}, "All splits required for deduplication"

    start_time = time.time()

    split_to_sketches = {}
    for sg_file in sg_files:
        split_name = get_split_name(sg_file)
        filter_filenames = split_to_filenames[split_name]
        sketches = convert_split(file=sg_file, split_name=split_name, filter_filenames=filter_filenames,
                                 limit=limit, quantize_bits=quantize_bits, new_tokens=new_tokens)
        split_to_sketches[split_name] = sketches

    split_to_sketches = deduplicate_splits(split_to_sketches)

    save_splits(output_dir, split_to_sketches)

    print(f"Processing Time: {time.time() - start_time} secs")

def main(sg_file, output_dir, limit, quantize_bits, new_tokens, split_name):
    sketches = convert_split(file=sg_file, split_name=split_name, 
                                limit=limit, quantize_bits=quantize_bits, new_tokens=new_tokens)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="/home/ubuntu/vitruvion/data/sg_strings",
                        help="Input folder containing the SketchGraphs filter sequence .npy files")
    parser.add_argument("--output", type=str, default='/home/ubuntu/sifan/data/sg_strings_v6_with_constraints/',
                        help="Output folder to save the data [default: output]")
    parser.add_argument("--filter", type=str, default="/home/ubuntu/sifan/cad_llm/split_to_filenames_v4.json",
                        help="File containing indices of deduped sketches ('train_test.json')")
    parser.add_argument("--limit", type=int, help="Only process this number of designs")
    parser.add_argument("--quantize_bits", type=int, default=6,
                        help="Number of bits to use for quantization (default 6)")
    parser.add_argument("--new_tokens", type=int, default=0,
                        help="Set to nonzero to use new token encoding")
    args = parser.parse_args()

    output_dir = get_output_dir(args)
    sg_files = get_files(args)

    filter_main(sg_files=sg_files, output_dir=output_dir, filter_path=args.filter, limit=args.limit,
         quantize_bits=args.quantize_bits, new_tokens=args.new_tokens)
