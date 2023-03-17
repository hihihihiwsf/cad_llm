"""
Train a CAD LLM model
"""


try:
    import comet_ml  # Import before torch
except ImportError:
    pass
from dataset.sg_dataset import get_sketchgraphs_dataloader
from models.byt5 import ByT5Model
from torch.utils.data import DataLoader
from util import get_loggers, get_checkpoint_callbacks
from args.main_args import get_training_args
from pathlib import Path
from pytorch_lightning import Trainer


def main():
    """Entry point for our training script"""
    args = get_training_args()

    results_dir = Path(args.results_dir) / args.exp_name
    if not results_dir.exists():
        results_dir.mkdir(parents=True)

    checkpoint_dir = Path(args.checkpoint_dir) / "checkpoints" / args.exp_name
    args.checkpoint_dir = str(checkpoint_dir)
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)

    loggers = get_loggers(args=args, log_dir=results_dir)

    print("Loading model...")
    model = ByT5Model(args=args)

    print("Loading data...")
    train_dataloader = get_sketchgraphs_dataloader(tokenizer=model.tokenizer, args=args, split="train", shuffle=False)
    val_dataloader = get_sketchgraphs_dataloader(tokenizer=model.tokenizer, args=args, split="val", shuffle=True)

    call_backs = get_checkpoint_callbacks(log_dir=results_dir, all_checkpoint_dir=checkpoint_dir,
                                          using_sagemaker=args.using_sagemaker)
    print("Training the model...")
    log_every_n_steps = 100
    trainer = Trainer(
        callbacks=call_backs,
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
        logger=loggers,
        max_epochs=3,
        log_every_n_steps=log_every_n_steps,
        val_check_interval=0.25,
        resume_from_checkpoint=None,
    )
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == "__main__":
    main()
