import json
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import aws_utils


def get_loggers(args, log_dir):
    """Get the loggers to use"""
    args_file = log_dir / "args.json"
    with open(args_file, "w", encoding="utf8") as f:
        json.dump(vars(args), f, indent=4)

    csv_logger = pl.loggers.CSVLogger(log_dir, name="log")
    tb_logger = pl.loggers.TensorBoardLogger(log_dir, name="tb_log")
    loggers = [csv_logger, tb_logger]
    comet_json_path = Path("cometml.json")
    if args.comet and comet_json_path.exists():
        with open(comet_json_path) as json_file:
            comet_config = json.load(json_file)
        # Creat a new comet experiment
        comet_logger = pl.loggers.CometLogger(
            api_key=comet_config["api_key"],
            workspace=comet_config["workspace"],
            project_name=comet_config["project_name"],
            experiment_name=args.exp_name,
            log_code=False,
            log_git_metadata=False,
            log_git_patch=False,
        )
        args.comet_experiment_key = comet_logger.experiment.get_key()
        loggers.append(comet_logger)
    return loggers


def get_checkpoint_callbacks(log_dir, all_checkpoint_dir, using_sagemaker):
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", dirpath=log_dir, filename=f"best",
                                          save_last=True)
    # Also save in a checkpoint directory that is backed up to s3 during training ??
    checkpoint_callback.CHECKPOINT_NAME_LAST = "last"
    # all_checkpoint_callback = ModelCheckpoint(dirpath=all_checkpoint_dir, filename="{epoch}", every_n_epochs=1)
    callbacks = [checkpoint_callback]
    if using_sagemaker:
        # Sync the checkpoints to s3 manually after each epoch
        callbacks.append(aws_utils.SyncCheckpoint())
    return callbacks


def get_quantized_range(quantize_n_bits):
    return range(-2 ** (quantize_n_bits - 1), 2 ** (quantize_n_bits - 1))

import numpy as np
class EmbeddingCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.image_embeddings = []
        self.txt_embeddings = []
        self.pixel = []
        self.name = []
        self.fulltext = []
        self.intext = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        #self.image_embeddings.append(pl_module.image_embeddings.detach().cpu().numpy())
        self.txt_embeddings.append(pl_module.txt_embeddings.detach().cpu().numpy())
        self.pixel.append(pl_module.px.detach().cpu().numpy())
        self.name.append(pl_module.name)
        self.fulltext.append(pl_module.fulltext)
        self.intext.append(pl_module.intext)