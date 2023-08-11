"""

AWS utility functions from:
https://docs.aws.amazon.com/sagemaker/latest/dg/distributed-troubleshooting-model-parallel.html

"""


import os
from pytorch_lightning.callbacks import Callback


class SyncCheckpoint(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        # Sync the checkpoints to Sagemaker manually
        if trainer.global_rank == 0 and pl_module.args.using_sagemaker:
            base_s3_uri = os.path.dirname(os.path.dirname(os.getenv("SM_MODULE_DIR", "")))
            full_s3_uri = f"{base_s3_uri}/checkpoints/"
            print(f"Syncing checkpoints from local {pl_module.args.checkpoint_dir} to s3 {full_s3_uri}")
            sync_local_to_s3(local_path=pl_module.args.checkpoint_dir, s3_uri=full_s3_uri)


class SyncSamples(Callback):
    def on_val_epoch_end(self, trainer, pl_module):
        # Sync the samples from Sagemaker to s3 manually
        if trainer.global_rank == 0 and pl_module.args.using_sagemaker:
            base_s3_uri = os.path.dirname(os.path.dirname(os.getenv("SM_MODULE_DIR", "")))
            full_s3_uri = f"{base_s3_uri}/samples/"
            print(f"Syncing checkpoints from local {pl_module.args.samples_dir} to s3 {full_s3_uri}")
            sync_local_to_s3(local_path=pl_module.args.samples_dir, s3_uri=full_s3_uri)


def aws_s3_sync(source, destination):
    """aws s3 sync in quiet mode and time profile"""
    import time
    import subprocess
    cmd = ["aws", "s3", "sync", "--quiet", source, destination]
    print(f"Syncing files from {source} to {destination}")
    start_time = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.wait()
    end_time = time.time()
    print("Time Taken to Sync: ", (end_time - start_time))
    return


def sync_local_to_s3(local_path, s3_uri):
    """ sample function to sync checkpoints from local path to s3 """

    import boto3
    # check if local path exists
    if not os.path.exists(local_path):
        raise RuntimeError("Provided local path {local_path} does not exist. Please check")

    # check if s3 bucket exists
    s3 = boto3.resource('s3')
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Provided s3 uri {s3_uri} is not valid.")

    s3_bucket = s3_uri.replace('s3://', '').split('/')[0]
    print(f"S3 Bucket: {s3_bucket}")
    try:
        s3.meta.client.head_bucket(Bucket=s3_bucket)
    except Exception as e:
        raise e
    aws_s3_sync(local_path, s3_uri)
    return
