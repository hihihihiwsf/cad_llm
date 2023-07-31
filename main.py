"""
Train a CAD LLM model
"""


try:
    import comet_ml  # Import before torch
except ImportError:
    pass
from dataset.sg_dataset import get_sketchgraphs_dataloader, SketchDataModule
from models.byt5 import ByT5Model
from torch.utils.data import DataLoader
from util import get_loggers, get_checkpoint_callbacks
from args.main_args import get_training_args
from pathlib import Path
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor
from models.segformer import SegformerModel
from dataset.rendered_sketch_dataset import get_rendered_sketch_dataset
from dataset.sketch_strings_dataset import get_sketch_strings_dataset
from dataset.sketch_strings_collator import SketchStringsCollator



def get_model(args):
    if "segformer" in args.model_name:
        return SegformerModel(model_name=args.model_name)

    return ByT5Model(args=args)


def get_dataloader(min_input_percent, max_input_percent, args, split, shuffle, model):
    if "segformer" in args.model_name:
        datasets = get_rendered_sketch_dataset(min_input_percent, max_input_percent, path=args.dataset)
        dataloader = DataLoader(datasets[split], batch_size=args.batch_size, shuffle=shuffle,
                                num_workers=args.num_workers)
        return dataloader

    if "entities" in args.dataset:  # Hack to select dataset loader based on dataset name
        datasets = get_sketch_strings_dataset(path=args.dataset, min_split_ratio=args.min_input_percent,
                                             max_split_ratio=args.max_input_percent)

        collator = SketchStringsCollator(tokenizer=model.tokenizer, max_length=args.max_length)

        if args.limit_data != 1 and split == "train":
            n = int(args.limit_data * len(datasets["train"]))
            datasets["train"] = datasets["train"].shuffle().select(range(n))

        return DataLoader(datasets[split], batch_size=args.batch_size, collate_fn=collator, shuffle=shuffle,
                          num_workers=args.num_workers)

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
    #train_dataloader = get_dataloader(min_input_percent, max_input_ercent,args=args, split="train", shuffle=True, model=model)
    # train_dataloader = []
    # for i in range(4):
    #     min_per = 0.65-i*0.15
    #     each_dataloader = get_dataloader(min_input_percent=min_per, max_input_percent=0.8, args=args, split="train", shuffle=True, model=model)
    #     train_dataloader.append(each_dataloader)
        
    #val_dataloader = get_dataloader(args=args, split="val", shuffle=False, model=model)

    #model.set_total_train_steps(num_train_batches=len(train_dataloader))

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
        reload_dataloaders_every_n_epochs=10,
        # resume_from_checkpoint=None,
        # check_val_every_n_epoch=args.val_every_n_epoch,
        val_check_interval=args.val_check_interval,
        # limit_train_batches=0.001,
        # limit_val_batches=0.01,
    )
    sketchdata = SketchDataModule(model.tokenizer, args)
    if not args.eval: 
        # for epoch in range(0,40,10):
        #     dataloader = train_dataloader[epoch//10]
        #trainer.fit(model, train_dataloaders=dataloader, val_dataloaders=val_dataloader)
        trainer.fit(model, datamodule=sketchdata)
    else:
        # loading the model from exp_name/best.ckpt
        ckpt_dir = args.checkpoint_dir + "/{}/best.ckpt".format(args.exp_name)
        trainer.validate(model, ckpt_path=ckpt_dir, datamodule=sketchdata)


if __name__ == "__main__":
    main()
