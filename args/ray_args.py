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
        default=3, 
        help="The number of CPUs to use per GPU"
    )
    parser.add_argument(
        "--strategy", 
        type=str, 
        default="ddp", 
        help="The name of distributed strategy used by lightening trainer ('ddp' or 'fsdp')"
    )
    parser.add_argument(
        "--input_s3_bucket", 
        type=str, 
        default="cad-llm-katzm/dataset/sg_strings_v5_ascii1_max64/", 
        help="Name of the input S3 bucket"
    )
    parser.add_argument(
        "--output_s3_bucket", 
        type=str, 
        default="ray-training-out-sifan", 
        help="Name of the output S3 bucket"
    )
    ray_args, _ = parser.parse_known_args()
    
    return ray_args
