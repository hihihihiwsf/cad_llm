"""
Train a CAD LLM model
"""


try:
    import comet_ml  # Import before torch
except ImportError:
    pass
from dataset.sg_dataset import get_sketchgraphs_dataloader
from models.gpt_neo import GPT_Neo
from torch.utils.data import DataLoader
from util import get_loggers, get_checkpoint_callbacks
from args.main_args import get_training_args
from pathlib import Path
# import pytorch_lightning as pl
import lightning.pytorch as pl

import torch
import os
from pytorch_lightning.strategies import DeepSpeedStrategy
from finetuning_scheduler import FinetuningScheduler
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint


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

    print("Loading model...")
    # model = ByT5Model(args=args)
    model = GPT_Neo(args=args)

    print("Loading data...")
    train_dataloader = get_sketchgraphs_dataloader(tokenizer=model.tokenizer, args=args, split="train", shuffle=True)
    val_dataloader = get_sketchgraphs_dataloader(tokenizer=model.tokenizer, args=args, split="val", shuffle=False)

    call_backs = get_checkpoint_callbacks(log_dir=results_dir, all_checkpoint_dir=checkpoint_dir,
                                          using_sagemaker=args.using_sagemaker)

    # call_backs = []
    # call_backs.append(LearningRateMonitor(logging_interval='step'))
    # call_backs = [LearningRateMonitor(logging_interval='step')]
    call_backs.append(FinetuningScheduler())
    # pl.utilities.seed.seed_everything(args.seed)

    print("Training the model...")

    log_every_n_steps = 100
    trainer = pl.Trainer(
        callbacks=call_backs,
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
        logger=loggers,
        max_epochs=args.epochs,
        log_every_n_steps=log_every_n_steps,
        # strategy=DeepSpeedStrategy(config="./ds_config_gpt_neo_27.json"),
        gradient_clip_val=2.,
        # accumulate_grad_batches=16,
        # precision=16,
        # resume_from_checkpoint=None,
        # limit_train_batches=0.001,
        # limit_val_batches=0.01
    )
    if not args.eval: 
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    else:
        # loading the model from exp_name/best.ckpt
        ckpt_dir = args.checkpoint_dir + "/{}/best.ckpt".format(args.exp_name)
        trainer.validate(model, ckpt_path=ckpt_dir, dataloaders=val_dataloader)


if __name__ == "__main__":
    # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    main()
