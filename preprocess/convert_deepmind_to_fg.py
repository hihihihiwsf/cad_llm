"""

Convert Deepmind sketch dataset sketches
into the Fusion 360 Gallery dataset format

"""

import argparse
import uuid
import json
from pathlib import Path
import numpy as np

from preprocess.deepmind_geometry import *
from preprocess.fusiongallery_geometry import *
from preprocess.preprocess_utils import get_files, get_output_dir



def convert_data(dm_data):
    for i, dm_sketch in enumerate(dm_data):
        fg_sketch = convert_sketch(dm_sketch)


def convert_sketch(dm_sketch):
    dm_entities = dm_sketch["entitySequence"]["entities"]
    dm_constraints = dm_sketch["constraintSequence"]["constraints"]
    points, point_map = create_sketch_points(dm_entities)
    curves, constraint_entity_map = create_sketch_curves(dm_entities, point_map)
    # TODO: Filter out sketches with no curves
    assert len(curves) > 0
    constraints = create_constraints(dm_constraints, points, curves, constraint_entity_map)
    # dimensions = create_dimensions()
    fusion_gallery_sketch = {
        "name": "Sketch1",
        "type": "Sketch",
        "points": points,
        "curves": curves,
        "constraints": {},
        "dimensions": {}
    }
    return fusion_gallery_sketch


def create_sketch_curves(dm_entities, point_map):
    """Create the sketch curves data structure"""
    # Curve dictionary as stored in FG json format
    curve_data = {}
    # Mapping from the constraint entities index array to uuids in either the points or curves dict
    constraint_entity_map = {}
    for index, dm_ent in enumerate(dm_entities):
        assert len(dm_ent) == 1, "Expected on entry in the dict"
        entity_name = list(dm_ent.keys())[0]
        entity_data = dm_ent[entity_name]

        if entity_name == "pointEntity":
            # Find the point in the point map
            fg_obj = FusionGalleryPoint.from_xy_map(
                entity_data["point"]["x"],
                entity_data["point"]["y"],
                point_map.map
            )
            constraint_entity_map[index] = {
                "type": "point",
                "uuid": fg_obj.uuid
            }
            # Continue here as we don't want to add points to the curve data
            continue

        elif entity_name == "lineEntity":
            dm_obj = DeepmindLine(entity_data)
            fg_obj = FusionGalleryLine(dm_obj, point_map.map)
        elif entity_name == "circleArcEntity":
            if "arcParams" in entity_data:
                dm_obj = DeepmindArc(entity_data)
                fg_obj = FusionGalleryArc(dm_obj, point_map.map)
            elif "circleParams" in entity_data:
                dm_obj = DeepmindCircle(entity_data)
                fg_obj = FusionGalleryCircle(dm_obj, point_map.map)
        else:
            print("Warning - unexpected entity name", entity_name)
            print(dm_ent)
            continue
        curve_data[fg_obj.uuid] = fg_obj.to_dict()
        constraint_entity_map[index] = {
            "type": "curve",
            "uuid": fg_obj.uuid
        }
    return curve_data, constraint_entity_map
        

def create_sketch_points(dm_entities):
    """Create the sketch points data structure"""
    # point_map.map containts a dict of unique points
    # key: string of the form point.x_point.y
    # value: FusionGalleryPoint object
    point_map = FusionGalleryPointMap(dm_entities)
    # Sketch points data structure in the FG format
    point_data = {}
    for key, fg_point in point_map.map.items():
        point_data[fg_point.uuid] = fg_point.to_dict()
    return point_data, point_map    


def create_constraints(dm_constraints, points, curves, constraint_entity_map):
    """Create the constraints data structure"""
    constraints_data = {}
    for dm_cst in dm_constraints:
        constraint = FusionGalleryConstraint(dm_cst, points, curves, constraint_entity_map)
        cst_dict = constraint.to_dict()
        if cst_dict is not None:
            constraints_data[constraint.uuid] = cst_dict
    return constraints_data

def main(args):

    input_path = Path(args.input)
    if input_path.is_dir():
        input_file_paths = get_files(args.input, pattern="*.json")
        input_file_paths.sort()
    elif input_path.is_file():
        input_file_paths = [input_path]

    output_path = get_output_dir(args.output)

    for input_file_path in input_file_paths:
        with open(input_file_path) as f:
            dm_data = json.load(f)
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
