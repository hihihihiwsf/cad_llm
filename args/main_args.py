import argparse
import multiprocessing
import os


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True,
                        help="Experiment name for lookup in experiment_log.py")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save checkpoints and logs")
    parser.add_argument("--dataset", type=str, default="data/sg_strings", help="Dataset path")
    parser.add_argument("--num_workers", type=int, default=-1, help="Number of workers to use in the torch dataloader")
    parser.add_argument("--comet", type=int, default=0, help="Use comet.ml for experiment tracking")
    parser.add_argument("--ascii_encoding", type=int, default=0,
                        help="Use ascii ByT5 encoding instead of single token encoding")
    return parser


def get_main_args_for_launch():
    """Get training args to send to SageMaker as hyperparameters"""
    parser = get_parser()
    args, _ = parser.parse_known_args()
    return args


def get_training_args():
    """Get the training args for training. Override some args if on sagemaker."""
    parser = get_parser()
    args = parser.parse_args()

    # Change flag int args (required by sagemaker) back to bool
    args.ascii_encoding = bool(args.ascii_encoding)
    args.comet = bool(args.comet)
    if args.num_workers == -1:
        args.num_workers = multiprocessing.cpu_count()

    # If using sagemaker override directory locations
    args.using_sagemaker = os.getenv("SM_MODEL_DIR") is not None
    args.results_dir = os.getenv("SM_MODEL_DIR") or args.results_dir
    args.dataset = os.getenv("SM_CHANNEL_TRAIN") or args.dataset
    args.checkpoint_dir = "/opt/ml" if args.using_sagemaker else args.results_dir

    return args
