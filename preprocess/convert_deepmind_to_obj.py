from deepmind_geometry import DeepmindLine, DeepmindArc, DeepmindCircle
from preprocessing import center_and_scale
import numpy as np
import json
from preprocess_utils import get_files, get_output_dir
import argparse


def convert_data(dm_data):
    obj_data = []

    for i, dm_sketch in enumerate(dm_data):
        obj_sketch = convert_sketch(dm_sketch)
        if not obj_sketch:
            continue
        obj_sketch["index"] = i
        obj_data.append(obj_sketch)

    return obj_data


def convert_sketch(dm_sketch):
    dm_entities = dm_sketch["entitySequence"]["entities"]

    # Return None if sketch contains splines
    if any(get_dm_ent_name(dm_ent) == "interpolatedSplineEntity" for dm_ent in dm_entities):
        return None

    # Remove single points
    dm_entities = [dm_ent for dm_ent in dm_entities if get_dm_ent_name(dm_ent) != "pointEntity"]

    # Parse entity dicts to entity objects
    entities = [parse_entity(dm_ent) for dm_ent in dm_entities]

    # Return None if sketch contains bad entities (e.g. 0 radius circles)
    if any(ent.exception for ent in entities):
        return None

    # Filter out construction entities and None entities (come from single points)
    entities = [ent for ent in entities if ent and not ent.is_construction]

    if not entities:
        return None

    # Convert entity classes to lists of points
    point_entities = [ent.to_points() for ent in entities]

    # Convert to obj format
    # Note: we normalize after removing construction entities so the bounding box may be smaller
    vertices, curves = convert_point_entities_to_obj(point_entities, bits=8)

    # Remove sketches with entities with duplicate points
    for curve in curves:
        if len(curve) > len(set(curve)):
            return None

    return {"vertices": vertices, "curves": curves}


def convert_point_entities_to_obj(point_entities, bits):
    """
    Return an obj like format of (vertices, curves)
    Vertex coordinates are normalized to [-1, 1]
    Note: coordinates are quantized to [-2**bits, 2**bits] for welding
    """
    quant_entities = [quantize_ent(ent, bits=bits) for ent in point_entities]
    vertices = []
    curves = []

    for quant_ent in quant_entities:
        curve = []
        for point in quant_ent:
            if point not in vertices:
                vertices.append(point)
            i = vertices.index(point)
            curve.append(i + 1)

        curves.append(curve)

    # Normalize to coordinates [-1, 1]
    vertices = center_and_scale(np.array(vertices)) * 2

    return vertices, curves


def quantize_ent(ent, bits):
    return [(quantize_float(x, bits=bits), quantize_float(y, bits=bits)) for x, y in ent]


def quantize_float(x, bits):
    """ Transfer float in range [-1, 1] to int in range [- 2 ** (bits - 1), 2 ** (bits - 1)]"""
    return int(x * (2**bits))


def parse_entity(dm_ent):
    assert len(dm_ent) == 1, "Expected on entry in the dict"
    entity_name = get_dm_ent_name(dm_ent)

    if entity_name == "pointEntity":
        return None

    if entity_name == "lineEntity":
        return DeepmindLine(dm_ent[entity_name])

    if entity_name == "circleArcEntity":
        if "arcParams" in dm_ent[entity_name]:
            return DeepmindArc(dm_ent[entity_name])
        elif "circleParams" in dm_ent[entity_name]:
            return DeepmindCircle(dm_ent[entity_name])

    print("Warning - unexpected entity name", entity_name)
    print(dm_ent)


def get_dm_ent_name(dm_ent):
    return list(dm_ent.keys())[0]


def load_data(path):
    with open(path) as f:
        data = json.load(f)
    return data


def main(args):
    input_file_paths = get_files(args.input, pattern="*.json")
    input_file_paths.sort()

    output_path = get_output_dir(args.output)

    for input_file_path in input_file_paths:
        dm_data = load_data(input_file_path)
        obj_data = convert_data(dm_data)

        filename = input_file_path.stem
        print(f"File {filename}: Converted {len(obj_data)} of {len(dm_data)} sketches to obj format")

        np.save(output_path / filename, obj_data, allow_pickle=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input folder containing DeepMind json files")
    parser.add_argument("--output", type=str, help="Output folder to save the obj-like json data [default: output]")
    args = parser.parse_args()

    main(args)
