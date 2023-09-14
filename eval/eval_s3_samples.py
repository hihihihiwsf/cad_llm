import argparse
import glob
import json
from collections import defaultdict

from adsk_ailab_ray.tools.aws import aws_s3_sync
from tqdm import tqdm

from cad_tokenizers.sketch_single_token_byt5_tokenizer import SketchSingleTokenByt5Tokenizer

from eval_metrics.top1_entity_metric import Top1EntityMetric
from eval_metrics.top1_full_sketch_metric import Top1FullSketchMetric
from eval_metrics.validity_metric import ValidityMetric
from pathlib import Path


def load_samples(base_path, epoch):
    val_name_to_samples = defaultdict(list)

    for cur_path in glob.iglob(base_path + f"/**/samples_epoch_{epoch}*.json", recursive=True):
        with open(cur_path) as f:
            cur_val_name_to_samples = json.load(f)

        for val_name, cur_samples in cur_val_name_to_samples.items():
            val_name_to_samples[val_name].extend(cur_samples)

    return val_name_to_samples


def get_samples(local_path, epoch, reprocess):
    process_samples_path = Path(local_path) / f"processed_samples_epoch_{epoch}.json"

    if process_samples_path.is_file() and not reprocess:
        print(f"Loading processed samples from {process_samples_path}")
        with open(process_samples_path) as f:
            processed_samples = json.load(f)
        return processed_samples

    print(f"Loading samples from {local_path}")
    samples = load_samples(local_path, epoch)

    print("Processing samples")
    processed_samples = process_samples_all_val_sets(val_name_to_sample_infos=samples)

    print(f"Saving {len(processed_samples)} processed samples to {str(process_samples_path)}")

    def set_default(obj):
        if isinstance(obj, set):
            return list(obj)
        return obj

    with open(process_samples_path, "w") as json_file:
        json.dump(processed_samples, json_file, default=set_default)

    return processed_samples


def process_samples_all_val_sets(val_name_to_sample_infos):
    val_name_to_processed = {}

    model_name = "google/byt5-small"
    tokenizer = SketchSingleTokenByt5Tokenizer.from_pretrained(model_name)

    for val_name, sample_infos in val_name_to_sample_infos.items():
        processed = process_samples(sample_infos=sample_infos, tokenizer=tokenizer)
        val_name_to_processed[val_name] = processed

    return val_name_to_processed


def process_samples(sample_infos, tokenizer):
    processed_samples = []

    seen = set()
    for info in tqdm(sample_infos):
        # Remove duplicates
        if info["name"] in seen:
            continue
        seen.add(info["name"])

        # Decode and parse to point entities
        pred_text = tokenizer.decode(info["samples"], skip_special_tokens=True)
        pred = tokenizer.new_tokens_str_to_entities(text=pred_text, sort=True)
        if "true" in info:
            true = info["true"]
        else:
            true = tokenizer.new_tokens_str_to_entities(info["output_text"], sort=True)

        pred = set(tuple(sorted([tuple(p) for p in ent])) for ent in pred if ent)
        true = set(tuple(sorted([tuple(p) for p in ent])) for ent in true if ent)

        info["pred_point_entities"] = pred
        info["true_point_entities"] = true

        processed_samples.append(info)

    return processed_samples


def compute_metrics_all_val_sets(val_name_to_samples):
    all_metrics = {}
    for val_name, samples in val_name_to_samples.items():
        res = compute_metrics(samples)
        all_metrics[val_name] = res

    return all_metrics


def compute_metrics(samples):
    top1_full_sketch = Top1FullSketchMetric()
    top1_entity = Top1EntityMetric()
    validity = ValidityMetric()

    for sample in samples:
        top1_entity.update(pred=sample["pred_point_entities"], true=sample["true_point_entities"])
        top1_full_sketch.update(pred=sample["pred_point_entities"], true=sample["true_point_entities"])
        validity.update(pred=sample["pred_point_entities"])

    return {
        "top1_full_sketch": top1_full_sketch.compute(),
        "top1_entity": top1_entity.compute(),
        "validity": validity.compute(),
    }


def main(local_path, s3_path, reprocess, epoch):
    if s3_path:
        aws_s3_sync(s3_path, local_path)

    samples = get_samples(local_path, epoch=epoch, reprocess=reprocess)

    print(f"epoch {epoch} ({len(samples)} samples)")
    metrics_result = compute_metrics_all_val_sets(samples)
    print(metrics_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_path", required=True, type=str, help="Local path to save samples and metrics")
    parser.add_argument("--s3_path", default=None, type=str, help="Remote s3 path to samples directory")
    parser.add_argument("--reprocess", default=False, type=int, help="Reprocess samples")
    parser.add_argument("--epoch", required=True, type=int, help="Get metrics for the specified epoch")
    args = parser.parse_args()

    main(local_path=args.local_path, s3_path=args.s3_path, reprocess=args.reprocess, epoch=args.epoch)
