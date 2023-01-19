"""

Convert the SketchGraphs Dataset sketches
into a npy file containing obj-file like data

"""

# hack for running locally
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import time
import argparse
import importlib
import json
import numpy as np
from tqdm import tqdm
from sketchgraphs.data import flat_array, sketch_from_sequence
import sketch_sg

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


def convert_sketch(index, seq, split_name, norm):
    """Convert a sketch from the SketchGraphs dataset"""
    sketch = sketch_from_sequence(seq)
    sketch_name = f"{split_name}_{index:08}"
    sketch_obj = SketchSG(sketch, sketch_name)
    return sketch_obj.convert(normalize=norm)


def convert_split(file, output_dir, split_name, filter_filenames, limit, norm):
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
    sketch_obj_dicts = []
    for filename in tqdm(filter_filenames[:seq_limit]):
        index = get_index(filename)
        seq = seq_data["sequences"][index]
        sketch_obj_dict = convert_sketch(index=index, seq=seq, split_name=split_name, norm=norm)
        sketch_obj_dicts.append(sketch_obj_dict)

    np.save(output_dir / f"sg_obj_{split_name}.npy", sketch_obj_dicts)


def main(sg_files, output_dir, filter_path, limit, norm):
    split_to_filenames = load_filter(filter_path)

    start_time = time.time()

    for sg_file in sg_files:
        split_name = get_split_name(sg_file)
        filter_filenames = split_to_filenames[split_name]
        convert_split(file=sg_file, output_dir=output_dir, split_name=split_name,
                      filter_filenames=filter_filenames, limit=limit, norm=norm)

    print(f"Processing Time: {time.time() - start_time} secs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Input folder containing the SketchGraphs filter sequence .npy files")
    parser.add_argument("--output", type=str, help="Output folder to save the data [default: output]")
    parser.add_argument("--filter", type=str, required=True,
                        help="File containing indices of deduped sketches ('train_test.json')")
    parser.add_argument("--limit", type=int, help="Only process this number of designs")
    parser.add_argument("--no-norm",  action='store_true', help="Disables normalization of vertices.")
    args = parser.parse_args()

    output_dir = get_output_dir(args)
    sg_files = get_files(args)

    main(sg_files=sg_files, output_dir=output_dir, filter_path=args.filter,
         limit=args.limit, norm=(not args.no_norm))
