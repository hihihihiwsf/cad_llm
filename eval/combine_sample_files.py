import argparse
import json
import os

def main(path, prefix):
    infos = read_infos(path, prefix)
    infos = deduplicate_infos(infos)
    save_infos(path, prefix, infos)


def read_infos(path, prefix):
    data = []

    filenames = [filename for filename in os.listdir(path) if filename.startswith(prefix)]
    for filename in filenames:
        cur_path = os.path.join(path, filename)
        with open(cur_path) as json_file:
            cur_data = json.load(json_file)
            data.extend(cur_data)

    return data


def deduplicate_infos(infos):
    seen = set()
    deduped_infos = []

    for info in infos:
        if info['name'] in seen:
            continue
        deduped_infos.append(info)
        seen.add(info['name'])

    duplicate_count = len(infos) - len(deduped_infos)
    print(f"Removed {duplicate_count} samples")

    return deduped_infos


def save_infos(path, prefix, infos):
    filename = prefix + "_all.json"
    path = os.path.join(path, filename)
    with open(path, "w") as json_file:
        json.dump(infos, json_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, type=str, help="Path to input sample files. Used for output as well.")
    parser.add_argument("--prefix", required=True, type=str, help="Prefix of files to combine")

    args = parser.parse_args()

    main(path=args.path, prefix=args.prefix)
