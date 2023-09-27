"""
Train a CAD LLM model on a Ray Cluster
"""

import time
from pathlib import Path

from adsk_ailab_ray.ray_lightning import RayLightningExperiment
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.loggers import CometLogger

from args.ray_args import get_ray_args
from dataset.byt5_datamodule import Byt5DataModule
from models.byt5_v2 import ByT5v2
from cad_tokenizers.cad_tokenizers_utils import get_tokenizer_cls


from functools import partial
import torch
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp import MixedPrecision
import json


def get_loggers(exp_name, use_comet, comet_workspace, comet_project_name):
    loggers = [CSVLogger("logs"), TensorBoardLogger("logs")]

    if use_comet:
        comet_logger = CometLogger(
            workspace=comet_workspace,
            project_name=comet_project_name,
            experiment_name=exp_name,
            log_code=False,
            log_git_metadata=False,
            log_git_patch=False,
        )
        loggers.append(comet_logger)

    return loggers


def train_on_ray_cluster():
    args = get_ray_args()

    exp_name = args.exp_name + "_" + time.strftime("%Y%m%d-%H%M%S")

    tokenizer_cls = get_tokenizer_cls(args.tokenizer_name)

    extra_val_percentages = [20, 40, 60, 80] if args.add_extra_val_sets else []
    val_names = Byt5DataModule.val_dataloader_names(extra_val_percentages)  # Send names to model for logging

    # Configure LightningModule and LightningDataModule classes and kwargs
    data_class = Byt5DataModule
    data_class_kwargs = {
        "model_name": args.model_name,
        "tokenizer_cls": tokenizer_cls,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "min_ratio": args.min_split_ratio,
        "max_ratio": args.max_split_ratio,
        "input_s3_bucket": args.input_s3_bucket,
        "dataset_path": args.local_dataset_dir,
        "num_dataloader_workers": min(args.num_dataloader_workers, args.num_cpus_per_worker),
        "extra_val_percentages": extra_val_percentages,
    }

    model_class = ByT5v2
    tokenizer = tokenizer_cls.from_pretrained(args.model_name)
    local_samples_path = Path(args.local_results_dir) / exp_name / "samples"
    remote_samples_path = f"s3://{args.output_s3_bucket}/{exp_name}/samples"
    model_class_kwargs = {
        "model_name": args.model_name,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "local_samples_path": local_samples_path,
        "remote_samples_path": remote_samples_path,
        "tokenizer": tokenizer,
        "val_names": val_names,
    }

    strategy_kwargs = {}
    if args.strategy == "fsdp":
        if args.mix_precision:
            strategy_kwargs["mixed_precision"] = MixedPrecision(param_dtype=torch.bfloat16, cast_forward_inputs=True)
        if args.cpu_offload:
            strategy_kwargs["cpu_offload"] = True
    elif args.strategy == "deepspeed":
        strategy_kwargs["stage"] = 2

    # Configure lightning trainer kwargs
    loggers = get_loggers(exp_name, args.comet, comet_workspace=args.comet_workspace,
                          comet_project_name=args.comet_project_name)
    trainer_kwargs = {
        "accumulate_grad_batches": args.grad_accu,
        "logger": loggers,
        "max_epochs": args.max_epochs,
        "accelerator": "auto",
        "log_every_n_steps": args.log_every_n_steps,
        "val_check_interval": args.val_check_interval,
    }

    if args.mix_precision:
        trainer_kwargs["precision"] = 'bf16'

    # Configure ray checkpointing kwargs
    checkpointing_kwargs = {
        "monitor": "val_loss",
        "mode": "min",
        "save_top_k": 1
    }

    # Define an Experiment
    experiment = RayLightningExperiment(
        exp_name=exp_name,
        timestamp_exp_name=False,
        num_workers=args.num_gpus,
        num_cpus_per_worker=args.num_cpus_per_worker,
        strategy=args.strategy,
        strategy_kwargs=strategy_kwargs,
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
        ckpt_path=args.ckpt_path,
    )

    # Run the experiment on the Ray cluster
    result = experiment.run()
    print("Validation Loss: ", result.metrics["val_loss"])


if __name__ == "__main__":
    train_on_ray_cluster()
