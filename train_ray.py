"""
Train a CAD LLM model on a Ray Cluster
"""

import time
from pathlib import Path

import torch
from adsk_ailab_ray.ray_lightning import RayLightningExperiment
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from torch.distributed.fsdp import MixedPrecision

from args.ray_args import get_ray_args
from cad_tokenizers.cad_tokenizers_utils import get_tokenizer_cls
from dataset.byt5_datamodule import Byt5DataModule
from models.byt5_v2 import ByT5v2
from models.llama2 import Llama2Model, get_model_bucket_uri, get_model_download_dir, get_model_checkpoint_and_refs_dir, download_model_weights
import json

def train_on_ray_cluster():
    args = get_ray_args()

    exp_name = args.exp_name + "_" + time.strftime("%Y%m%d-%H%M%S")

    model_class = Llama2Model
    # if "llama" in args.model_name.lower():
    model_id = args.model_name
    model_bucket_uri = get_model_bucket_uri(model_id)
    model_download_dir = get_model_download_dir(model_id)
    model_checkpoint_path, _ = get_model_checkpoint_and_refs_dir(model_id)

    download_model_weights(model_id, model_bucket_uri, model_download_dir)

    model_config = Llama2Model.get_config(model_checkpoint_path)
    tokenizer = Llama2Model.get_tokenizer(model_checkpoint_path)
    tokenizer.pad_token = tokenizer.eos_token
    print('TOKEEEEEENENNNNNIIIIIZZZER'*100, tokenizer)
    
    
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
        "s3_data_uri": args.s3_data_uri,
        "dataset_path": args.local_dataset_dir,
        "num_dataloader_workers": min(args.num_dataloader_workers, args.num_cpus_per_worker),
        "extra_val_percentages": extra_val_percentages,
    }


        
    tokenizer = tokenizer_cls.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    SPECIAL_TOKENS = ["<SYSTEM>", "<START_Q>", "<END_Q>", "<START_A>", "<END_A>"]
    tokenizer.add_tokens(SPECIAL_TOKENS, special_tokens=True)
    
    
    local_samples_path = Path(args.local_results_dir) / exp_name / "samples"
    remote_samples_path = f"{args.s3_results_uri}/{exp_name}/samples"
    model_class_kwargs = {
        "model_name": args.model_name,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "local_samples_path": local_samples_path,
        "remote_samples_path": remote_samples_path,
        "tokenizer": tokenizer,
        "val_names": val_names,
        "model_bucket_uri": model_bucket_uri,
        "model_download_dir": model_download_dir,
        "model_checkpoint_path": model_checkpoint_path,
        "vocab_size": len(tokenizer),
    }

    strategy_kwargs = {}
    if args.strategy == "fsdp":
        if args.mix_precision:
            strategy_kwargs["mixed_precision"] = MixedPrecision(param_dtype=torch.bfloat16, cast_forward_inputs=True)
        if args.cpu_offload:
            strategy_kwargs["cpu_offload"] = True
    elif args.strategy == "deepspeed":
        deepspeed_config_path = str(Path(__file__).resolve().parent / 'deepspeed_configs/zero_3_llama_2_7b.json')
        with open(deepspeed_config_path, 'r') as f:
            deepspeed_config = json.loads(f.read())

        # strategy_kwargs = {"config": deepspeed_config}
        strategy_kwargs["stage"] = 2

    # Configure lightning trainer kwargs
    loggers = [CSVLogger("logs"), TensorBoardLogger("logs")]  # Comet is integrated into RayLightningExperiment

    trainer_kwargs = {
        "accumulate_grad_batches": args.grad_accu,
        "logger": loggers,
        "max_epochs": args.max_epochs,
        "accelerator": "auto",
        "log_every_n_steps": args.log_every_n_steps,
        "val_check_interval": args.val_check_interval,
        "devices": "1" if args.dry_run else "auto",
        "limit_train_batches":0.005,
        "limit_val_batches": 0.01
    }

    if args.mix_precision:
        trainer_kwargs["precision"] = 'bf16'

    # Configure ray checkpointing kwargs
    checkpointing_kwargs = {
        "monitor": "val_loss",
        "mode": "min",
        "save_top_k": 1
    }

    comet_experiment_kwargs = {
        "workspace": args.comet_workspace,
        "project_name": args.comet_project_name,
        "experiment_key": args.comet_experiment_key,
        "log_code": False,
        "log_git_metadata": False,
        "log_git_patch": False
    }

    # print('WE ARE BEFORE MODULE' * 100)
    # asd = Byt5DataModule(**data_class_kwargs)
    # dm = asd.get_dataset()
    # print('DS SIZE' * 20)
    # print(len(dm['train']))
    # print('WE ARE AFTER MODULE' * 100)
    
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
        s3_data_uri=args.s3_data_uri,
        s3_results_uri=args.s3_results_uri,
        local_data_dir=args.local_dataset_dir,
        local_results_dir=args.local_results_dir,
        max_failures=args.max_failures,
        ckpt_path=args.ckpt_path,
        worker_node_type=args.worker_node_type,
        worker_node_life_cycle=args.worker_node_life_cycle,
        comet_experiment_kwargs=comet_experiment_kwargs
    )

    # Run the experiment on the Ray cluster
    if args.dry_run:
        result = experiment.fit_dry_run()
    else:
        result = experiment.fit()
        
    print("Validation Loss: ", result.metrics["val_loss"])


if __name__ == "__main__":
    train_on_ray_cluster()
