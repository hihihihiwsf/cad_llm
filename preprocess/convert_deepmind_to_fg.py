"""

Convert Deepmind sketch dataset sketches
into the Fusion 360 Gallery dataset format

"""

from deepmind_geometry import DeepmindLine, DeepmindArc, DeepmindCircle, DeepmindPoint
from fusiongallery_geometry import FusionGalleryPoint
import numpy as np
import json
from pathlib import Path
import uuid
from preprocess_utils import get_files, get_output_dir
from convert_deepmind_to_obj import load_data, get_dm_ent_name
import argparse



def convert_data(dm_data):
    for i, dm_sketch in enumerate(dm_data):
        convert_sketch(dm_sketch)


def convert_sketch(dm_sketch):
    dm_entities = dm_sketch["entitySequence"]["entities"]
    points, point_map = create_sketch_points(dm_entities)
    # curves = create_sketch_curves()
    # constraints = create_constraints()
    # dimensions = create_dimensions()
    # fusion_gallery_sketch = {
    #     "name": "Sketch1",
    #     "type": "Sketch",
    #     "points": points,
    #     "curves": curves,
    #     "constraints": constraints,
    #     "dimensions": dimensions
    # }

    # return fusion_gallery_sketch


def create_sketch_curves(dm_entities):
    """Create the sketch curves data structure"""
    curve_data = {}
    for dm_ent in dm_entities:
        assert len(dm_ent) == 1, "Expected on entry in the dict"
        entity_name = get_dm_ent_name(dm_ent)
        entity_data = dm_ent[entity_name]

        if entity_name == "pointEntity":
            continue
        elif entity_name == "lineEntity":
            dm_obj = DeepmindLine(entity_data)
        elif entity_name == "circleArcEntity":
            if "arcParams" in entity_data:
                dm_obj = DeepmindArc(entity_data)
            elif "circleParams" in entity_data:
                dm_obj = DeepmindCircle(entity_data)
        else:
            print("Warning - unexpected entity name", entity_name)
            print(dm_ent)
            continue

    curve_data[dm_obj.uuid] = dm_obj.to_fg_dict()
    return curve_data
        

def create_sketch_points(dm_entities):
    """Create the sketch points data structure"""
    # Dict of unique points
    # key: string of the form point.x_point.y
    # value: FusionGalleryPoint object
    point_map = get_point_map(dm_entities)
    # Sketch points data structure in the FG format
    point_data = {}
    for key, fg_point in point_map.items():
        point_data[fg_point.uuid] = fg_point.to_dict()
    return point_data, point_map


def get_point_map(dm_entities):
    """Get a dictionary of unique points used by the sketch entities"""
    point_dict = {}
    point_count = 0
    for dm_ent in dm_entities:
        points = find_all_geom_points(dm_ent)
        for point in points:
            point_count += 1
            # First make a deepmind point then pass that to our fusion gallery point
            dm_point = DeepmindPoint(point)
            fg_point = FusionGalleryPoint(dm_point)
            if not fg_point.key in point_dict:
                point_dict[fg_point.key] = fg_point
    print(f"Created vertex dictionary with {len(point_dict)} of {point_count} total vertices")
    return point_dict


def find_all_geom_points(dm_ent):
    """Given an OnShape entity, return all of its points"""
    assert len(dm_ent) == 1, "Expected on entry in the dict"
    entity_name = get_dm_ent_name(dm_ent)
    entity_data = dm_ent[entity_name]

    if entity_name == "pointEntity":
        return [entity_data["point"]]
    if entity_name == "lineEntity":
        return [entity_data["start"], entity_data["end"]]
    if entity_name == "circleArcEntity":
        if "arcParams" in entity_data:
            return [
                entity_data["center"],
                entity_data["arcParams"]["start"],
                entity_data["arcParams"]["end"]
            ]
        elif "circleParams" in entity_data:
            return [entity_data["center"]]            


def main(args):

    input_path = Path(args.input)
    if input_path.is_dir():
        input_file_paths = get_files(args.input, pattern="*.json")
        input_file_paths.sort()
    elif input_path.is_file():
        input_file_paths = [input_path]

    output_path = get_output_dir(args.output)

    for input_file_path in input_file_paths:
        dm_data = load_data(input_file_path)
        convert_data(dm_data)

        # filename = input_file_path.stem
        # print(f"File {filename}: Converted {len(obj_data)} of {len(dm_data)} sketches to obj format")

        # np.save(output_path / filename, obj_data, allow_pickle=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input file/folder of DeepMind json files")
    parser.add_argument("--output", type=str, help="Output folder to save the Fusion 360 Gallery json data [default: output]")
    args = parser.parse_args()

    main(args)
