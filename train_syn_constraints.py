"""
Train a CAD LLM model on a Ray Cluster
"""

from pytorch_lightning.callbacks import ModelCheckpoint

from dataset.syn_constraints_dataset import SynConstraintsDataModule
from models.byt5_syn_constraints import ByT5SynConstraintsModel

try:
    import comet_ml  # Import before torch
except ImportError:
    pass
from util import get_loggers, get_checkpoint_callbacks
from args.main_args import get_training_args
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor


def main():
    """Entry point for our training script"""
    args = get_training_args()

    results_dir = Path(args.results_dir) / args.exp_name
    if not results_dir.exists():
        results_dir.mkdir(parents=True)

    samples_dir = results_dir / "samples"
    args.samples_dir = str(samples_dir)
    if not samples_dir.exists():
        samples_dir.mkdir()

    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)

    loggers = get_loggers(args=args, log_dir=results_dir)

    pl.seed_everything(args.seed)

    datamodule = SynConstraintsDataModule(args=args, ray_args=None)
    datamodule.setup(stage="fit")
    tokenizer = datamodule.get_tokenizer()

    call_backs = get_checkpoint_callbacks(log_dir=results_dir, all_checkpoint_dir=checkpoint_dir,
                                          using_sagemaker=args.using_sagemaker)
    call_backs.append(LearningRateMonitor(logging_interval='step'))
    log_every_n_steps = 100

    model = ByT5SynConstraintsModel(args=args, tokenizer=tokenizer)
    trainer = pl.Trainer(
        callbacks=call_backs,
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
        logger=loggers,
        max_epochs=args.epochs,
        log_every_n_steps=log_every_n_steps,
        val_check_interval=args.val_check_interval,
        limit_train_batches=0.001,
        limit_val_batches=0.01,
    )

    print("Training the model...")
    trainer.fit(model, train_dataloaders=datamodule.train_dataloader(), val_dataloaders=datamodule.val_dataloader())


if __name__ == "__main__":
    main()
