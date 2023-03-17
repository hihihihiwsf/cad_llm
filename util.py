import json
from pathlib import Path
import pytorch_lightning as pl
from experiment_log import experiment_name_to_hps
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


def  get_checkpoint_callbacks(log_dir, all_checkpoint_dir, using_sagemaker):
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", dirpath=log_dir, filename=f"best",
                                          save_last=True)
    # Also save in a checkpoint directory that is backed up to s3 during training ??
    checkpoint_callback.CHECKPOINT_NAME_LAST = "last"
    all_checkpoint_callback = ModelCheckpoint(dirpath=all_checkpoint_dir, filename="{epoch}", every_n_epochs=1)
    callbacks = [checkpoint_callback, all_checkpoint_callback]
    if using_sagemaker:
        # Sync the checkpoints to s3 manually after each epoch
        callbacks.append(aws_utils.SyncCheckpoint())
    return callbacks


def get_exp_hyperparams(exp_name, log_dir):
    print(f"Loading hyperparameters for experiment '{exp_name}'")
    hyperparams = experiment_name_to_hps[exp_name]
    hyperparams_file = log_dir / "hyperparams.json"
    # Save the hyperparameters
    print(hyperparams)
    with open(hyperparams_file, "w", encoding="utf8") as f:
        json.dump(hyperparams, f, indent=4)
    return hyperparams


def get_quantized_range(quantize_n_bits):
    return range(-2 ** (quantize_n_bits - 1), 2 ** (quantize_n_bits - 1))
