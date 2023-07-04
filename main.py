"""
Train a CAD LLM model
"""


try:
    import comet_ml  # Import before torch
except ImportError:
    pass
from dataset.sg_dataset_vl import get_sketchgraphs_dataloader
#from models.byt5 import ByT5Model
from models.blip import BLIPModel
from torch.utils.data import DataLoader
from util import get_loggers, get_checkpoint_callbacks
from args.main_args import get_training_args
from pathlib import Path
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor
from models.segformer import SegformerModel
from dataset.vertex_grid_dataset import get_vertex_grid_dataset


def get_model(args):
    if "segformer" in args.model_name:
        return SegformerModel()

    return BLIPModel(args=args)


def get_dataloader(args, split, shuffle, model):
    if "segformer" in args.model_name:
        datasets = get_vertex_grid_dataset(path=args.dataset)
        train_dataloader = DataLoader(datasets[split], batch_size=args.batch_size, shuffle=shuffle,
                                      num_workers=args.num_workers)
        return train_dataloader

    return get_sketchgraphs_dataloader(tokenizer=model.tokenizer, args=args, split=split, shuffle=shuffle)


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

    # pl.utilities.seed.seed_everything(args.seed)
    pl.seed_everything(args.seed)

    print("Loading model...")

    model = get_model(args)

    print("Loading data...")
    train_dataloader = get_dataloader(args=args, split="train", shuffle=True, model=model)
    val_dataloader = get_dataloader(args=args, split="val", shuffle=False, model=model)

    model.set_total_train_steps(num_train_batches=len(train_dataloader))

    call_backs = get_checkpoint_callbacks(log_dir=results_dir, all_checkpoint_dir=checkpoint_dir,
                                          using_sagemaker=args.using_sagemaker)

    call_backs.append(LearningRateMonitor(logging_interval='step'))

    print("Training the model...")
    log_every_n_steps = 10000

    world_size = torch.cuda.device_count()
    trainer = pl.Trainer(
        callbacks=call_backs,
        accelerator="gpu", #args.accelerator,
        devices=world_size,
        strategy="fsdp",
        precision=16,
        logger=loggers,
        max_epochs=args.epochs,
        log_every_n_steps=log_every_n_steps,
        # resume_from_checkpoint=None,
        # check_val_every_n_epoch=args.val_every_n_epoch,
        val_check_interval=args.val_check_interval,
        # limit_train_batches=0.001,
        # limit_val_batches=0.01,
    )
    if not args.eval: 
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    else:
        # loading the model from exp_name/best.ckpt
        ckpt_dir = args.checkpoint_dir + "/{}/best.ckpt".format(args.exp_name)
        trainer.validate(model, ckpt_path=ckpt_dir, dataloaders=val_dataloader)


if __name__ == "__main__":
    main()
