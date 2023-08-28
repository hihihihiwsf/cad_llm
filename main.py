"""
Train a CAD LLM model
"""


try:
    import comet_ml  # Import before torch
except ImportError:
    pass
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.utils.data import DataLoader

from args.main_args import get_training_args
from dataset.rendered_sketch_dataset import get_rendered_sketch_dataset
from dataset.sg_dataset import get_sketchgraphs_dataloader
from dataset.sketch_strings_collator import SketchStringsCollator
from dataset.sketch_strings_dataset import get_sketch_strings_dataset
from models.byt5 import ByT5Model
from models.segformer import SegformerModel
from util import get_loggers, get_checkpoint_callbacks
from dataset.syn_constraints_dataset import (
    SynConstraintsDataModule,
    SynConstraintsPPDataModule,
    SynConstraintsSchema2DataModule,
)
from models.byt5_syn_constraints import ByT5SynConstraintsModel


def get_model(args, tokenizer, total_train_steps):
    if "syn_constraints" in args.model_name:
        return ByT5SynConstraintsModel(model_name="google/byt5-small", lr=args.lr, batch_size=args.batch_size,
                                       max_length=args.max_length, checkpoint_dir=args.checkpoint_dir,
                                       samples_dir=args.samples_dir, tokenizer=tokenizer, use_adafactor=args.adafactor)

    if "segformer" in args.model_name:
        return SegformerModel(model_name=args.model_name, checkpoint_dir=args.checkpoint_dir)

    if "byt5" in args.model_name:
        return ByT5Model(args=args, tokenizer=tokenizer, total_train_steps=total_train_steps)

    raise ValueError(f"Unsupported model type {args.model_name}")


def get_dataloader_and_tokenizer(args):
    if "syn_constraints" in args.model_name:
        if args.model_name == "syn_constraints":
            model_class = SynConstraintsDataModule
        elif args.model_name == "syn_constraints_pp":
            model_class = SynConstraintsPPDataModule
        elif args.model_name == "syn_constraints_schema2":
            model_class = SynConstraintsSchema2DataModule

        datamodule = model_class(model_name="google/byt5-small", batch_size=args.batch_size, max_length=args.max_length,
                                 dataset_path=args.dataset, num_workers=args.num_workers)

        datamodule.setup(stage="fit")
        tokenizer = datamodule.get_tokenizer()
        return {"train": datamodule.train_dataloader(), "val": datamodule.val_dataloader()}, tokenizer

    if "segformer" in args.model_name:
        datasets = get_rendered_sketch_dataset(path=args.dataset)
        train_dataloader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_workers)
        val_dataloader = DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers)
        return {"train": train_dataloader, "val": val_dataloader}, None

    if "entities" in args.dataset:  # Hack to select dataset loader based on dataset name
        tokenizer = ByT5Model.get_tokenizer(args.model_name)

        datasets = get_sketch_strings_dataset(path=args.dataset, min_split_ratio=args.min_input_percent,
                                              max_split_ratio=args.max_input_percent)

        collator = SketchStringsCollator(tokenizer=tokenizer, max_length=args.max_length)

        if args.limit_data != 1:
            n = int(args.limit_data * len(datasets["train"]))
            datasets["train"] = datasets["train"].shuffle().select(range(n))

        train_dataloader = DataLoader(datasets["train"], batch_size=args.batch_size, collate_fn=collator, shuffle=True,
                                      num_workers=args.num_workers)
        val_dataloader = DataLoader(datasets["val"], batch_size=args.batch_size, collate_fn=collator, shuffle=False,
                                    num_workers=args.num_workers)
        return {"train": train_dataloader, "val": val_dataloader}, tokenizer
    else:
        tokenizer = ByT5Model.get_tokenizer(args.model_name)

        train_dataloader = get_sketchgraphs_dataloader(tokenizer=tokenizer, args=args, split="train", shuffle=True)
        val_dataloader = get_sketchgraphs_dataloader(tokenizer=tokenizer, args=args, split="val", shuffle=False)
        return {"train": train_dataloader, "val": val_dataloader}, tokenizer


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

    print("Loading data...")
    dataloader, tokenizer = get_dataloader_and_tokenizer(args=args)

    num_train_batches = len(dataloader["train"])
    num_gpus = torch.cuda.device_count()
    total_train_steps = ByT5Model.get_total_train_steps(num_train_batches, num_gpus, args.epochs)

    print("Loading model...")
    model = get_model(args, tokenizer=tokenizer, total_train_steps=total_train_steps)

    call_backs = get_checkpoint_callbacks(log_dir=results_dir, all_checkpoint_dir=checkpoint_dir,
                                          using_sagemaker=args.using_sagemaker)

    call_backs.append(LearningRateMonitor(logging_interval='step'))

    print("Training the model...")
    log_every_n_steps = 10000
    trainer = pl.Trainer(
        callbacks=call_backs,
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
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
        trainer.fit(model, train_dataloaders=dataloader["train"], val_dataloaders=dataloader["val"])
    else:
        # loading the model from exp_name/best.ckpt
        # TODO: get this working on sagemaker
        ckpt_dir = args.checkpoint_dir + "/{}/best.ckpt".format(args.exp_name)
        trainer.validate(model, ckpt_path=ckpt_dir, dataloaders=dataloader["val"])


if __name__ == "__main__":
    main()
