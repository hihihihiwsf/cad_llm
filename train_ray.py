"""
Train a CAD LLM model on a Ray Cluster
"""

from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers.csv_logs import CSVLogger

from adsk_ailab_ray.ray_lightning import RayLightningExperiment

from util import get_checkpoint_callbacks
from dataset.sg_dataset import get_sketchgraphs_dataloader
from dataset.sg_dataset import SketchGraphsDataModule
from models.byt5 import ByT5Model
from models.vl_t5 import VLT5Model
from models.vision_only import VisionT5Model
from models.vl_biloss import BiVLT5Model

from models.vis_recon import VisRecon


from args.main_args import get_main_args_for_launch
from args.ray_args import get_ray_args


def get_dataloader(args, split, shuffle, model):

    return get_sketchgraphs_dataloader(tokenizer=model.tokenizer, args=args, split=split, shuffle=shuffle)


def train_on_ray_cluster():

    ray_args = get_ray_args()
    main_args = get_main_args_for_launch()  # args to pass to main.py

    main_args.samples_dir = main_args.results_dir

    loggers = [CSVLogger("logs")]

    #model = ByT5Model(args=main_args)
    model = ByT5Model(args=main_args, vit_mae=None)
    
    datamodule = SketchGraphsDataModule(
        tokenizer=model.tokenizer,
        args=main_args,
        ray_args=ray_args
    )
    # Prepere dataset to get the number of train batches
    datamodule.setup("fit")
    num_train_batches = len(datamodule.train_dataloader())

    ByT5Model.set_total_train_steps_ray(
        num_train_batches=num_train_batches,
        n_gpus=ray_args.num_gpus,
        epochs=main_args.epochs
    )

    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", dirpath="checkpoints", filename=f"best",
                                          save_last=True)
    checkpoint_callback.CHECKPOINT_NAME_LAST = "last"
    call_backs = [checkpoint_callback]

    call_backs.append(LearningRateMonitor(logging_interval='step'))

    log_every_n_steps = 10000

    # Set your LightningModule and LightningDataModule classes
    model_class = ByT5Model
    data_class = SketchGraphsDataModule

    # Configure hyperparameters and other settings
    exp_name = main_args.exp_name
    num_workers = ray_args.num_gpus
    num_cpus_per_worker = ray_args.num_cpus_per_worker
    strategy = ray_args.strategy
    model_class_kwargs = {"args": main_args}
    data_class_kwargs = {
        "tokenizer": model.tokenizer,
        "args": main_args,
        "ray_args": ray_args
    }
    
    trainer_kwargs = {
            "callbacks": call_backs,
            "logger": loggers,
            "max_epochs": main_args.epochs,
            "accelerator": "auto",
            "log_every_n_steps": log_every_n_steps,
            "val_check_interval": main_args.val_check_interval,
    }
    checkpointing_kwargs = {
        "monitor": "val_loss",
        "mode": "min",
        "save_top_k": 1
    }
    input_s3_bucket = ray_args.input_s3_bucket
    output_s3_bucket = ray_args.output_s3_bucket

    local_data_dir = main_args.dataset
    local_results_dir = main_args.results_dir

    # Define an Experiment
    experiment = RayLightningExperiment(
            exp_name,
            num_workers,
            num_cpus_per_worker,
            strategy,
            model_class,
            model_class_kwargs,
            data_class,
            data_class_kwargs,
            trainer_kwargs,
            checkpointing_kwargs,
            input_s3_bucket,
            output_s3_bucket,
            local_data_dir,
            local_results_dir
    )

    # Run the experiment on the Ray cluster
    result = experiment.run()
    print("Validation Loss: ", result.metrics["val_loss"])


if __name__ == "__main__":
    train_on_ray_cluster()