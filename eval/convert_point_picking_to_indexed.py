import argparse
import json

import numpy as np

from eval.find_closest_lines import find_closest_lines
from preprocess.syn_contraints_preprocess import pp_constraints_from_string
from tqdm import tqdm


def main(input_path, output_path, dataset_path):
    infos = read(input_path)
    ds = read(dataset_path)
    infos = convert_infos_from_pp_to_indexed(infos, ds)
    save_infos(output_path, infos)


def read(path):
    with open(path) as json_file:
        data = json.load(json_file)
    return data


def convert_infos_from_pp_to_indexed(infos, ds):
    name_to_sketch = {sketch['filename']: sketch for sketch in ds['data']}

    for info in tqdm(infos):
        text_sample = info['text_sample']
        sketch = name_to_sketch[info['name']]
        vertices = np.array(sketch['vertices'])
        edges = sketch['edges']

        pp_constraints = pp_constraints_from_string(text_sample)
        indexed_constraints = convert_pp_constraints_to_indexed(pp_constraints, vertices, edges)

        info['true'] = sketch['constraints']
        info['pred'] = indexed_constraints


    return infos


def save_infos(path, data):
    with open(path, "w") as json_file:
        json.dump(data, json_file)


def points_to_indices(vertices, edges, points):
    indices = find_closest_lines(vertices, edges, points)
    indices = [int(i) for i in indices]
    return indices


def convert_pp_constraints_to_indexed(pp_constraints, vertices, edges):
    horizontal = points_to_indices(vertices, edges, pp_constraints["horizontal"])
    vertical = points_to_indices(vertices, edges, pp_constraints["vertical"])
    parallel = [points_to_indices(vertices, edges, points) for points in pp_constraints["parallel"]]
    perpendicular = [points_to_indices(vertices, edges, points) for points in pp_constraints["perpendicular"]]

    indexed_constraints = {
        "horizontal": list(set(horizontal)),
        "vertical": list(set(vertical)),
        "parallel": [list(set(indices)) for indices in parallel],
        "perpendicular": [indices for indices in perpendicular],
    }

    return indexed_constraints


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, type=str, help="Path to input sample files in point picking format")
    parser.add_argument("--dataset_path", required=True, type=str, help="Path to original dataset in obj format")
    parser.add_argument("--output_path", required=True, type=str, help="Path to output converted file in indexed format")

    args = parser.parse_args()

    main(input_path=args.input_path, output_path=args.output_path, dataset_path=args.dataset_path)
