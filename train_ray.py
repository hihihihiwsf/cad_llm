"""
Train a CAD LLM model on a Ray Cluster
"""

from pathlib import Path

from adsk_ailab_ray.ray_lightning import RayLightningExperiment
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

from args.ray_args import get_ray_args
from dataset.byt5_new_tokens_dataset import Byt5NewTokensDataModule
from models.byt5_v2 import ByT5v2
from util import get_comet_logger


def get_loggers(exp_name, use_comet):
    loggers = [CSVLogger("logs"), TensorBoardLogger("logs")]

    comet_logger = get_comet_logger(exp_name) if use_comet else None
    if comet_logger:
        loggers.append(comet_logger)

    return loggers


def train_on_ray_cluster():
    args = get_ray_args()

    # Configure LightningModule and LightningDataModule classes and kwargs
    data_class = Byt5NewTokensDataModule
    data_class_kwargs = {
        "model_name": args.model_name,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "min_ratio": args.min_split_ratio,
        "max_ratio": args.max_split_ratio,
        "input_s3_bucket": args.input_s3_bucket,
        "dataset_path": args.local_dataset_dir,
        "num_dataloader_workers": min(args.num_dataloader_workers, args.num_cpus_per_worker),
    }

    model_class = ByT5v2
    tokenizer = Byt5NewTokensDataModule.get_tokenizer(args.model_name)
    local_samples_path = Path(args.local_results_dir) / args.exp_name / "samples"
    remote_samples_path = f"s3://{args.output_s3_bucket}/{args.exp_name}/samples"
    model_class_kwargs = {
        "model_name": args.model_name,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "local_samples_path": local_samples_path,
        "remote_samples_path": remote_samples_path,
        "tokenizer_length": len(tokenizer),
    }

    # Configure lightning trainer kwargs
    loggers = get_loggers(args.exp_name, args.comet)
    trainer_kwargs = {
        "logger": loggers,
        "max_epochs": args.max_epochs,
        "accelerator": "auto",
    }

    # Configure ray checkpointing kwargs
    checkpointing_kwargs = {
        "monitor": "val_loss",
        "mode": "min",
        "save_top_k": 1
    }

    # Define an Experiment
    experiment = RayLightningExperiment(
        exp_name=args.exp_name,
        num_workers=args.num_gpus,
        num_cpus_per_worker=args.num_cpus_per_worker,
        strategy=args.strategy,
        strategy_kwargs={},
        model_class=model_class,
        model_class_kwargs=model_class_kwargs,
        data_class=data_class,
        data_class_kwargs=data_class_kwargs,
        trainer_kwargs=trainer_kwargs,
        checkpointing_kwargs=checkpointing_kwargs,
        input_s3_bucket=args.input_s3_bucket,
        output_s3_bucket=args.output_s3_bucket,
        local_data_dir=args.local_dataset_dir,
        local_results_dir=args.local_results_dir,
        max_failures=args.max_failures,
    )

    # Run the experiment on the Ray cluster
    result = experiment.run()
    print("Validation Loss: ", result.metrics["val_loss"])


if __name__ == "__main__":
    train_on_ray_cluster()
