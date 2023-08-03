"""

Convert SketchGraphs Dataset sketches
into the Fusion 360 Gallery dataset format

Input is the sg_all.npy file containing construction sequences
as a single file in a custom binary format.
https://github.com/PrincetonLIPS/SketchGraphs
Direct download link:
https://sketchgraphs.cs.princeton.edu/sequence/sg_all.npy

"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm

from preprocess.converter_base import ConverterBase
import preprocess.sketchgraphs.data as datalib
from preprocess.sketchgraphs.data import flat_array, sequence
from preprocess.sketchgraphs.data._entity import EntityType
from preprocess.sketchgraphs.data._constraint import ConstraintType, ConstraintParameterType, DirectionValue
from preprocess.sketchgraphs.pipeline import numerical_parameters


class SketchGraphsToFusionGalleryConverter(ConverterBase):
    """
    Class to handle conversion of SketchGraphs sketches
    into the Fusion 360 Gallery dataset format
    """    
    def __init__(self, input_file, output_path, limit):
        super().__init__()
        self.input_file = input_file
        self.output_path = output_path
        self.limit = limit

    def convert(self):
        """Convert the sketches"""
        print(f"Loading {self.input_file.name}...")
        seq_data = flat_array.load_dictionary_flat(self.input_file)
        sequences = seq_data["sequences"]
        print("Converting sketches...")
        for index, seq in enumerate(tqdm(sequences)):
            self.convert_sketch(index, seq)
            if self.limit is not None and self.converted_count >= self.limit:
                break


    def convert_sketch(self, index, seq):
        try:
            sketchgraphs_sketch = datalib.sketch_from_sequence(seq)


            # Save the file with the name of the original deepmind data file
            # and the index into that file of the sketch
            json_file = self.output_path / f"{index:08d}.json"
            # with open(json_file, "w") as f:
            #     json.dump(fg_sketch, f, indent=4)
            # if json_file.exists():
            #     self.converted_count += 1
            self.count += 1

        except Exception as ex:
            return None


def main(args):
    input_file = Path(args.input)
    output_path = Path(args.output)
    if not output_path.exists():
        output_path.mkdir()    
    converter = SketchGraphsToFusionGalleryConverter(input_file, output_path, args.limit)
    converter.convert()
    converter.print_log_results()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input file/folder of DeepMind json files")
    parser.add_argument("--output", type=str, help="Output folder to save the Fusion 360 Gallery json data [default: output]")
    parser.add_argument("--limit", type=int, help="Limit the number of files to process")
    args = parser.parse_args()
    main(args)
