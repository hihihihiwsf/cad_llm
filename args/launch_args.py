"""

Sagemaker args for launch.py

"""

import argparse


def get_launch_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aws_name",
        type=str,
        required=True,
        help="Username to prepend to aws job name"
    )
    parser.add_argument(
        "--aws_account",
        type=str,
        default="415862386602",
        help="AWS acount number used to set the role"
    )
    parser.add_argument(
        "--s3_bucket",
        type=str,
        default="cad-llm-katzm",
        help="s3 bucket name"
    )
    parser.add_argument(
        "--s3_dataset",
        type=str,
        default="sifan_test_ascii",
        help="Name of the dataset folder to use in s3"
    )
    parser.add_argument(
        "--instance_type",
        type=str,
        choices=("ml.p3.2xlarge", "ml.p3.8xlarge", "ml.p3.16xlarge", "ml.p3dn.24xlarge", "ml.p4d.24xlarge"),
        default="ml.p3.2xlarge",
        help="Sagemaker instance type",
    )
    parser.add_argument(
        "--instance_count",
        type=int,
        default=1,
        help="Number of Sagemaker instances to use",
    )
    parser.add_argument(
        "--use_spot_instances",
        type=bool,
        default=False,
        help="Train using AWS spot instances",
    )
    sagemaker_args, _ = parser.parse_known_args()
    return sagemaker_args
