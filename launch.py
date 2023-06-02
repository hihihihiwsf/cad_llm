"""
Launch a distributed training job on Sagemaker by passing on arguments to main.py
"""


import os
from datetime import datetime
import json
import sagemaker
import boto3
from sagemaker.pytorch import PyTorch
from args.launch_args import get_launch_args
from args.main_args import get_main_args_for_launch


def launch_sagemaker():
    """Launch a training job using AWS Sagemaker"""

    launch_args = get_launch_args()
    main_args = get_main_args_for_launch()  # args to pass to main.py

    aws_region = os.getenv("AWS_REGION")
    boto_session = boto3.session.Session(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
        region_name=aws_region,
    )
    sagemaker_session = sagemaker.Session(boto_session=boto_session)

    if launch_args.aws_account:
        role = f"arn:aws:iam::{launch_args.aws_account}:role/mintSageMakerExecutionRole"
    else:
        role = sagemaker.get_execution_role(sagemaker_session)

    gpu_counts = {
        "ml.p3.2xlarge": 1,
        "ml.p3.8xlarge": 4,
        "ml.p3.16xlarge": 8,
        "ml.p3dn.24xlarge": 8,
        "ml.p4d.24xlarge": 8,
    }
    processes_per_host = gpu_counts[launch_args.instance_type]

    print("Distributed training with a total of:")
    print(f"\t{processes_per_host * launch_args.instance_count} workers")
    print(f"\t{launch_args.instance_count} instances of {launch_args.instance_type}")
    print(f"\t{processes_per_host} GPU(s) per instance")

    entry_point = f"main.py"
    exp_name = main_args.exp_name

    exp_run_name = exp_name.replace("_", "-") + '-' + datetime.now().strftime("%m-%d-%y-%H%M")
    job_name = f"amir-{exp_run_name}"
    output_path = f"s3://{launch_args.s3_bucket}/jobs"
    print("Job name:", job_name)
    print("Entry point:", entry_point)
    print("Output path:", output_path)

    # Maximum runtime in seconds
    # set this to the account max of 432000 sec / 5 days
    max_run = 5 * 24 * 60 * 60
    # Max wait time for spot instances
    # This appears to have to be >= max_run
    max_wait = 5 * 24 * 60 * 60 if launch_args.use_spot_instances else None

    hyperparameters = vars(main_args)
    print("Hyperparameters:")
    print(json.dumps(hyperparameters, indent=4, sort_keys=True))

    estimator = PyTorch(
        entry_point=entry_point,
        source_dir=".",
        output_path=output_path + "/",
        code_location=output_path,
        role=role,
        sagemaker_session=sagemaker_session,
        instance_count=launch_args.instance_count,
        instance_type=launch_args.instance_type,
        volume_size=500,  # Joint data size alone is 22 GB
        framework_version='2.0',
        py_version='py39',
        hyperparameters=hyperparameters,
        max_run=max_run,
        # checkpoint_s3_uri=checkpoint_s3_uri,
        image_uri='763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.0.0-gpu-py310-cu118-ubuntu20.04-sagemaker',
        use_spot_instances=launch_args.use_spot_instances,
        max_wait=max_wait,
        debugger_hook_config=False  # Disable debugger
    )

    # We currently just use a single folder for train/valid/test, lets call it train
    dataset_path = {
        "train": f"s3://{launch_args.s3_bucket}/dataset/{launch_args.s3_dataset}"
    }
    estimator.fit(
        dataset_path,
        job_name=job_name,
        wait=False
    )
    print(f"Dispatched job: {job_name}")
    print(f"https://{aws_region}.console.aws.amazon.com/sagemaker/home?region={aws_region}#/jobs/{job_name}")
    return estimator


if __name__ == "__main__":
    launch_sagemaker()
