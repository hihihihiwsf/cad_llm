"""

AWS utility functions from:
https://docs.aws.amazon.com/sagemaker/latest/dg/distributed-troubleshooting-model-parallel.html

"""


import os
from pytorch_lightning.callbacks import Callback
from adsk_ailab_ray.tools.aws import aws_s3_sync

class SyncCheckpoint(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        # Sync the checkpoints to Sagemaker manually
        if trainer.global_rank == 0:
            base_s3_uri = os.path.dirname(os.path.dirname(os.getenv("SM_MODULE_DIR", "")))
            full_s3_uri = f"{base_s3_uri}/checkpoints/"
            print(f"Syncing checkpoints from local {pl_module.checkpoint_dir} to s3 {full_s3_uri}")
            sync_local_checkpoints_to_s3(local_path=pl_module.checkpoint_dir, s3_uri=full_s3_uri)


def sync_local_checkpoints_to_s3(local_path="/opt/ml/checkpoints", s3_uri=os.path.dirname(os.path.dirname(os.getenv('SM_MODULE_DIR', ''))) + '/checkpoints'):
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


def sync_s3_checkpoints_to_local(local_path="/opt/ml/checkpoints", s3_uri=os.path.dirname(os.path.dirname(os.getenv('SM_MODULE_DIR', ''))) + '/checkpoints'):
    """ sample function to sync checkpoints from s3 to local path """

    import boto3
    # try to create local path if it does not exist
    if not os.path.exists(local_path):
        print(f"Provided local path {local_path} does not exist. Creating...")
        try:
            os.makedirs(local_path)
        except Exception as e:
            raise RuntimeError(f"Failed to create {local_path}")

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
    aws_s3_sync(s3_uri, local_path)
    return
