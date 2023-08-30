import argparse
import multiprocessing
import os


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name for file names")
    parser.add_argument("--results_dir", type=str, default="/home/ubuntu/sifan/results", help="Directory to save checkpoints and logs")
    parser.add_argument("--dataset", type=str, default="/home/ubuntu/sifan/data/sg_string_v5_ascii_max64", help="Dataset path")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of workers to use in the torch dataloader")
    parser.add_argument("--accelerator", type=str, default="auto", help="Lightning Trainer accelerator parameter")
    parser.add_argument("--devices", type=str, default="auto", help="Lightning Trainer devices parameter")
    parser.add_argument("--strategy", type=str, default="ddp", help="Lightning Trainer strategy parameter")
    parser.add_argument("--comet", type=int, default=0, help="Use comet.ml for experiment tracking")
    parser.add_argument("--ascii_encoding", type=int, default=1,
                        help="Use ascii ByT5 encoding instead of single token encoding")
    parser.add_argument("--model_name", type=str, default="Salesforce/Codet5p-770m", help="Huggingface model name")
    parser.add_argument("--untrained_model", type=int, default=0, help="Use an untrained model")
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate")
    parser.add_argument("--batch_size", type=int, default=6, help="Number of sketches in a batch")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--min_input_percent", type=float, default=0.2,
                        help="Minimal percentage of sketch entities to choose as input")
    parser.add_argument("--max_input_percent", type=float, default=0.8,
                        help="Maximal percentage of sketch entities to choose as input")
    parser.add_argument("--max_length", type=int, default=192,
                        help="Maximal length in tokens for both input and output. Longer sequences will be truncated.")
    parser.add_argument("--train_order", type=str, default="sorted", choices=("sorted", "user", "random"),
                        help="Choose between sorted/user order for entities in the sketch")
    parser.add_argument("--eval", type=int, default=0, help="if true, goes directly to the eval mode. loading best from the ckpt dir")
    parser.add_argument("--seed", type=int, default=0, help="Random seed to use")
    parser.add_argument("--limit_data", type=float, default=1.0, help="Percentage of data to train on")
    # parser.add_argument("--val_every_n_epoch", type=int, default=1, help="Check validation after n training epochs")
    parser.add_argument("--val_check_interval", type=float, default=1., help="Check validation after fraction of epoch")
    parser.add_argument("--lora", type=int, default=0, help="Apply LoRA if true")
    parser.add_argument("--cosinedecay", type=int, default=1, help="Apply Cosine Learning rate decay if true")

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
    args.comet = bool(args.comet)
    args.eval = bool(args.eval)
    args.ascii_encoding = bool(args.ascii_encoding)
    args.lora = bool(args.lora)

    if args.num_workers == -1:
        args.num_workers = multiprocessing.cpu_count()
    args.strategy = None if args.strategy == "None" else args.strategy
    args.untrained_model = bool(args.untrained_model)

    # If using sagemaker override directory locations
    args.using_sagemaker = os.getenv("SM_MODEL_DIR") is not None
    args.results_dir = os.getenv("SM_MODEL_DIR") or args.results_dir
    args.dataset = os.getenv("SM_CHANNEL_TRAIN") or args.dataset
    args.checkpoint_dir = "/opt/ml" if args.using_sagemaker else args.results_dir

    return args
