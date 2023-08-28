import argparse
import json

from preprocess.syn_contraints_preprocess import pp_constraints_from_string, process_for_syn_constraints
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

        res = process_for_syn_constraints(sketch=sketch, return_mid_points=True)
        mid_points = res['mid_points']

        info['true'] = sketch['constraints']
        info['pred'] = pp_constraints_from_string(text_sample, mid_points=mid_points)

    return infos


def save_infos(path, data):
    with open(path, "w") as json_file:
        json.dump(data, json_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, type=str, help="Path to input sample files in point picking format")
    parser.add_argument("--dataset_path", required=True, type=str, help="Path to original dataset in obj format")
    parser.add_argument("--output_path", required=True, type=str, help="Path to output converted file in indexed format")

    args = parser.parse_args()

    main(input_path=args.input_path, output_path=args.output_path, dataset_path=args.dataset_path)
