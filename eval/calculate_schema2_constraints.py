import argparse
import json

from preprocess.syn_contraints_preprocess import constraints_from_string_schema2
from tqdm import tqdm
from dataset.syn_constraints_dataset import SynConstraintsSchema2DataModule


def main(input_path, output_path):
    infos = read(input_path)

    tokenizer = SynConstraintsSchema2DataModule.get_tokenizer()

    for info in tqdm(infos):
        text_sample = tokenizer.decode(info["samples"], skip_special_tokens=True)
        predicted_constraints = constraints_from_string_schema2(text_sample)

        info['true'] = info['constraints']
        info['pred'] = predicted_constraints

    save_infos(output_path, infos)


def read(path):
    with open(path) as json_file:
        data = json.load(json_file)
    return data


def save_infos(path, data):
    with open(path, "w") as json_file:
        json.dump(data, json_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, type=str, help="Path to input sample files in point picking format")
    parser.add_argument("--output_path", required=True, type=str, help="Path to output converted file in indexed format")

    args = parser.parse_args()

    main(input_path=args.input_path, output_path=args.output_path)
