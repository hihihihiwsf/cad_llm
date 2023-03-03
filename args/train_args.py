import argparse
import os


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True,
                        help="Experiment name for lookup in experiment_log.py")
    parser.add_argument("--dataset", type=str, default="dataset/sg_strings", help="Dataset path")
    parser.add_argument("--chkpt", type=str, default="results/checkpoints",
                        help="Output folder to save checkpoints (loading not implemented)")
    # Note: use int flag since we cannot send bool to sagemaker
    parser.add_argument("--parallel", type=int, default=0, help="Calling 4 gpus if set")
    return parser


def get_launching_args():
    """Get training args to send to SageMaker as hyperparameters"""
    parser = get_parser()
    args, _ = parser.parse_known_args()
    return args


def get_training_args():
    """Get the training args for training. Override some args if on sagemaker."""
    parser = get_parser()
    args = parser.parse_args()
    set_sagemaker_args(args)
    # Change flag int args (required by sagemaker) back to bool
    args.parallel = bool(args.parallel)
    return args


def set_sagemaker_args(args):
    # If using sagemaker update the dataset source and model output dir
    sagemaker_channel_train = os.getenv("SM_CHANNEL_TRAIN")
    sagemaker_model_dir = os.getenv("SM_MODEL_DIR")
    if sagemaker_channel_train is not None:
        args.dataset = sagemaker_channel_train
    if sagemaker_model_dir is not None:
        args.using_sagemaker = True
        args.exp_dir = sagemaker_model_dir
        # Set the checkpoint directory to the default used by sagemaker
        # args.checkpoint_dir = "/opt/ml/checkpoints"
    else:
        args.using_sagemaker = False
