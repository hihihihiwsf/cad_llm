"""

Ray args for train_ray.py

"""

import argparse


def get_ray_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name",
        type=str, 
        default="test-ray-ptl", 
        help="Experiment name to use for folder naming"
    )
    parser.add_argument(
        "--num_gpus", 
        type=int, 
        default=1, 
        help="Number of GPUs to train on."
    )
    parser.add_argument(
        "--num_cpus_per_worker", 
        type=int, 
        default=6,
        help="The number of CPUs to use per GPU"
    )
    parser.add_argument(
        "--strategy", 
        type=str, 
        default="ddp", 
        help="The name of distributed strategy used by lightening trainer ('ddp' or 'fsdp')"
    )
    parser.add_argument(
        "--cpu_offload",
        type=int,
        default=0,
        help="This specifies whether to offload parameters to CPU when not involved in computation. 1 for True, 0 for False"
    )
    parser.add_argument(
        "--mix_precision",
        type=int,
        default=0,
        help="Use mixed precision training. 1 for True, 0 for False"
    )
    parser.add_argument(
        "--max_failures",
        type=int,
        default="0",
        help="The maximum number of attempts to recover a run"
    )
    parser.add_argument(
        "--input_s3_bucket",
        type=str, 
        default="cad-llm-katzm/dataset/sg_entities_v5",
        help="Name of the input S3 bucket"
    )
    parser.add_argument(
        "--output_s3_bucket", 
        type=str, 
        default="cad-llm-katzm/ray-training-output",
        help="Name of the output S3 bucket"
    )
    parser.add_argument(
        "--local_results_dir",
        type=str,
        default="/home/ray/ray_results",
        help="Local directory to save checkpoints and logs"
    )
    parser.add_argument(
        "--local_dataset_dir",
        type=str,
        default="/home/ray/data",
        help="Local directory to save dataset"
    )
    parser.add_argument(
        "--comet",
        type=int,
        default=0,
        help="Use comet.ml for experiment tracking"
    )
    parser.add_argument(
        "--comet_workspace",
        type=str,
        default="cad_llm",
        help="Comet workspace"
    )
    parser.add_argument(
        "--comet_project_name",
        type=str,
        default="byt5-ray",
        help="Comet project name"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/byt5-small",
        help="Huggingface model name"
    )
    parser.add_argument(
        "--num_dataloader_workers",
        type=int,
        default=4,
        help="Number of workers to use in the torch dataloader"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Initial learning rate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of sketches in a batch"
    )
    parser.add_argument(
        "--min_split_ratio",
        type=float,
        default=0.2,
        help="Minimal ratio of sketch entities to choose as input"
    )
    parser.add_argument(
        "--max_split_ratio",
        type=float,
        default=0.8,
        help="Maximal ratio of sketch entities to choose as input"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=192,
        help="Maximal length in tokens for both input and output. Longer sequences will be truncated."
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=3,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path/URL/S3-Path of the checkpoint from which training is resumed."
    )

    ray_args, _ = parser.parse_known_args()
    
    return ray_args
