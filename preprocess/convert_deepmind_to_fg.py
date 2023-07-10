"""

Convert Deepmind sketch dataset sketches
into the Fusion 360 Gallery dataset format

The Deepmind sketch dataset comes in a protobuf format.
Before using this script either convert from protobuf to json using:
    preprocess/convertprotobuf.py
or get the converted json files from someone who has already
suffered through this process and taken one for the team.

"""

import argparse
import json
import uuid
from pathlib import Path
from tqdm import tqdm

from preprocess.deepmind_geometry import *
from preprocess.fusiongallery_geometry import *
from preprocess.preprocess_utils import get_files, get_output_dir
from tests.test_fusiongallery_sketch import TestFusionGallerySketch


def convert_data(dm_data, output_path, input_file, limit, count, converted_count):
    """Convert all the sketches in a data file and save them as individual json files"""
    for index, dm_sketch in enumerate(tqdm(dm_data)):
        fg_sketch = convert_sketch(dm_sketch, count)
        if fg_sketch is None:
            print(f"Skipping sketch {count}")
        else:
            # Save the file with the name of the original deepmind data file
            # and the index into that file of the sketch
            json_file = output_path / f"{input_file.stem}_{index:06d}.json"
            with open(json_file, "w") as f:
                json.dump(fg_sketch, f, indent=4)
            if json_file.exists():
                converted_count += 1
        count += 1
        if converted_count >= limit:
            break        
    return count, converted_count


def convert_sketch(dm_sketch, count):
    """Convert a single sketch in the deepmind sketch format into a Fusion Gallery sketch dict"""
    dm_entities = dm_sketch["entitySequence"]["entities"]
    dm_constraints = dm_sketch["constraintSequence"]["constraints"]
    # First check that the sketch is good to convert
    if not is_sketch_good(dm_entities):
        return None
    try:
        points, point_map = create_sketch_points(dm_entities)
        curves, constraint_entity_map = create_sketch_curves(dm_entities, point_map)
        constraints, dimensions = create_constraints_and_dimensions(dm_constraints, points, curves, constraint_entity_map)
        fusion_gallery_sketch = {
            "name": f"Sketch{count}", # Save the sketch with the overall index in the dataset
            "type": "Sketch",
            "points": points,
            "curves": curves,
            "constraints": constraints,
            "dimensions": dimensions
        }
    except Exception as ex:
        print("Sketch conversion failed", ex)
        return None
    sketch_valid = is_converted_sketch_valid(fusion_gallery_sketch)
    if not sketch_valid:
        return None
    return fusion_gallery_sketch


def is_sketch_good(dm_entities):
    """Check that a sketch is valid and supported conversion"""
    curve_count = 0
    for dm_ent in dm_entities:
        entity_type = list(dm_ent.keys())[0]
        if entity_type == "pointEntity":
            # Skip
            pass
        elif entity_type == "lineEntity":
            curve_count += 1
        elif entity_type == "circleArcEntity":
            curve_count += 1
        elif entity_type == "interpolatedSplineEntity":
            print("Warning - interpolatedSplineEntity not supported")
            return False
        else:
            print("Warning - unexpected entity type", entity_type)
            return False
    if curve_count == 0:
        print("Warning - sketch doesn't have any curves")
        return False
    return True


def is_converted_sketch_valid(sketch):
    """Test the converted sketch for self-consistency"""
    try:
        sketch_test = TestFusionGallerySketch()
        sketch_test.run_sketch_test(sketch)
        return True
    except Exception as ex:
        print("Converted sketch failed tests", ex)
        return False


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


def create_sketch_curves(dm_entities, point_map):
    """Create the sketch curves data structure"""
    # Curve dictionary as stored in FG json format
    curve_data = {}
    # Mapping from the constraint entities index array to uuids in either the points or curves dict
    constraint_entity_map = {}
    constraint_entity_index = 0
    for dm_ent in dm_entities:
        assert len(dm_ent) == 1, "Expected on entry in the dict"
        entity_type = list(dm_ent.keys())[0]
        entity_data = dm_ent[entity_type]
        add_curve = False

        if entity_type == "pointEntity":
            dm_obj = DeepmindPoint(entity_data)
            # Find the point in the point map
            fg_obj = FusionGalleryPoint.from_xy_map(
                dm_obj.x,
                dm_obj.y,
                point_map.map
            )

        elif entity_type == "lineEntity":
            dm_obj = DeepmindLine(entity_data)
            fg_obj = FusionGalleryLine(dm_obj, point_map.map)
            add_curve = True

        elif entity_type == "circleArcEntity":
            if "arcParams" in entity_data:
                dm_obj = DeepmindArc(entity_data)
                fg_obj = FusionGalleryArc(dm_obj, point_map.map)
                add_curve = True
            elif "circleParams" in entity_data:
                dm_obj = DeepmindCircle(entity_data)
                fg_obj = FusionGalleryCircle(dm_obj, point_map.map)
                add_curve = True

        # Generate the FG dict that will get saved to json for the entity
        fg_dict = fg_obj.to_dict()
        # Register the entity so we can map between the entity indices given in the deepmind
        # data and the FG curve and point dicts
        constraint_entity_map, constraint_entity_index = update_constraint_entity_map(
            constraint_entity_map,
            fg_obj,
            fg_dict,
            constraint_entity_index
        )
        # Add the entity to the FG curve dict if it is a line, arc, circle
        if add_curve:
            curve_data[fg_obj.uuid] = fg_dict

    return curve_data, constraint_entity_map
        

def update_constraint_entity_map(entity_map, fg_obj, fg_dict, index):
    """
    Update the given entity in the constraint entity map

    The key in the entity_map is an index counted based on a specific layout:

    entity0.part0, entity0.part1, ..., entity0.partN, entity1.part0, entity1.part1, ...

    The following entity types and parts are used in the layout

    Entity type                | Parts
    ---------------------------+-------------------------------------------------------------
    point_entity               | entity
    interpolated_spline_entity | entity, interp_point0, ..., interp_pointM, start_derivative,
                               |     end_derivative, start_point, end_point
    circle_arc_entity          | entity, center, [start_point, end_point] (if not closed)
    line_entity                | entity, start_point, end_point

    See: https://github.com/deepmind/deepmind-research/issues/377#issuecomment-1218204488
    
    """
    updated_index = index

    if fg_dict["type"] == "Point3D":
        # entity
        entity_map[updated_index] = {
            "type": "point",
            "uuid": fg_obj.uuid
        }
        updated_index += 1
    
    elif fg_dict["type"] == "SketchCircle":
        # entity
        entity_map[updated_index] = {
            "type": "curve",
            "uuid": fg_obj.uuid
        }
        updated_index += 1
        # center
        entity_map[updated_index] = {
            "type": "point",
            "uuid": fg_dict["center_point"],
            "parent": fg_obj.uuid
        }
        updated_index += 1

    elif fg_dict["type"] == "SketchArc":
        # entity
        entity_map[updated_index] = {
            "type": "curve",
            "uuid": fg_obj.uuid
        }
        updated_index += 1
        # center
        entity_map[updated_index] = {
            "type": "point",
            "uuid": fg_dict["center_point"],
            "parent": fg_obj.uuid
        }
        updated_index += 1
        # start_point
        entity_map[updated_index] = {
            "type": "point",
            "uuid": fg_dict["start_point"],
            "parent": fg_obj.uuid
        }
        updated_index += 1
        # end_point
        entity_map[updated_index] = {
            "type": "point",
            "uuid": fg_dict["end_point"],
            "parent": fg_obj.uuid
        }
        updated_index += 1
    
    elif fg_dict["type"] == "SketchLine":
        # entity
        # Store references to the start and end points
        entity_map[updated_index] = {
            "type": "curve",
            "uuid": fg_obj.uuid,
            "start_point": fg_dict["start_point"],
            "end_point": fg_dict["end_point"],
        }
        updated_index += 1
        # start_point
        entity_map[updated_index] = {
            "type": "point",
            "uuid": fg_dict["start_point"],
            "parent": fg_obj.uuid
        }
        updated_index += 1
        # end_point
        entity_map[updated_index] = {
            "type": "point",
            "uuid": fg_dict["end_point"],
            "parent": fg_obj.uuid
        }
        updated_index += 1
    
    else:
        print("Warning - unexpected entity type", fg_dict["type"])

    return entity_map, updated_index


def create_constraints_and_dimensions(dm_constraints, points, curves, constraint_entity_map):
    """Create the constraints and dimensions data structure"""
    constraints_data = {}
    dimensions_data = {}
    for constraint in dm_constraints:
        if FusionGalleryDimension.is_dimension(constraint):
            create_dimension(constraint, points, curves, constraint_entity_map, dimensions_data)
        else:
            create_constraint(constraint, points, curves, constraint_entity_map, constraints_data)
    return constraints_data, dimensions_data


def create_constraint(constraint, points, curves, constraint_entity_map, constraints_data):
    """Create a constraint and add it to the provided constraints_data dictionary"""
    constraint = FusionGalleryConstraint(constraint, points, curves, constraint_entity_map)
    cst_dict_or_list = constraint.to_dict()
    if cst_dict_or_list is not None:
        # Single constraint
        if isinstance(cst_dict_or_list, dict):
            constraints_data[constraint.uuid] = cst_dict_or_list
        # Multiple constraint case
        elif isinstance(cst_dict_or_list, list):
            for cst_dict in cst_dict_or_list:
                if cst_dict is not None:
                    cst_uuid = str(uuid.uuid1())
                    constraints_data[cst_uuid] = cst_dict


def create_dimension(dimension, points, curves, constraint_entity_map, dimensions_data):
    """Create a constraint and add it to the provided constraints_data dictionary"""
    dimension = FusionGalleryDimension(dimension, points, curves, constraint_entity_map)
    dimension_dict = dimension.to_dict()
    if dimension_dict is not None:
        dimensions_data[dimension.uuid] = dimension_dict


def main(args):

    input_path = Path(args.input)
    if input_path.is_dir():
        input_files = get_files(args.input, pattern="*.json")
        input_files.sort()
    elif input_path.is_file():
        input_files = [input_path]

    output_path = get_output_dir(args.output)

    # The data comes in 96 different data files
    # that we loop over here
    count = 0
    converted_count = 0
    for i, input_file in enumerate(input_files):
        print(f"Converting {i}/{len(input_files)} data files")
        with open(input_file) as f:
            dm_data = json.load(f)
        count, converted_count = convert_data(dm_data, output_path, input_file, args.limit, count, converted_count)
        if converted_count >= args.limit:
            break
    
    print(f"Converted {converted_count}/{count} sketches!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input file/folder of DeepMind json files")
    parser.add_argument("--output", type=str, help="Output folder to save the Fusion 360 Gallery json data [default: output]")
    parser.add_argument("--limit", type=int, help="Limit the number of files to process")
    args = parser.parse_args()

    main(args)
