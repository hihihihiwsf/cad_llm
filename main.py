"""
Train a CAD LLM model
"""


try:
    import comet_ml  # Import before torch
except ImportError:
    pass
from dataset.sg_dataset import SketchGraphsDataset, SketchGraphsCollator
from models.byt5 import ByT5Model
from torch.utils.data import DataLoader
from util import get_loggers, get_exp_hyperparams
from args.train_args import get_training_args
from pathlib import Path
from pytorch_lightning import Trainer


def main():
    """Entry point for our training script"""
    args = get_training_args()

    results_dir = Path(args.results_dir) / args.exp_name
    if not results_dir.exists():
        results_dir.mkdir(parents=True)

    hps = get_exp_hyperparams(exp_name=args.exp_name, log_dir=results_dir)
    loggers = get_loggers(args=args, log_dir=results_dir)

    print("Loading model...")
    model = ByT5Model(model_name=hps['modal'], checkpoint=None, no_pretrain=hps.get("no_pretrain", False))

    print("Loading data...")
    dataset_dir = Path(args.dataset)
    train_dataset = SketchGraphsDataset(dataset_dir=dataset_dir, split="train", subset_range=hps.get("subset_range"))
    val_dataset = SketchGraphsDataset(dataset_dir=dataset_dir, split="val", subset_range=hps.get("subset_range"))
    data_collator = SketchGraphsCollator(tokenizer=model.tokenizer, max_length=hps.get("max_length"))
    train_dataloader = DataLoader(train_dataset, batch_size=hps['batch_size'], collate_fn=data_collator, shuffle=True,
                                  num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=hps['batch_size'], collate_fn=data_collator, shuffle=False,
                                num_workers=args.num_workers)

    print("Training the model...")
    log_every_n_steps = 100
    trainer = Trainer(callbacks=None, accelerator="auto", devices="auto", logger=loggers, max_epochs=1,
                      log_every_n_steps=log_every_n_steps, resume_from_checkpoint=None)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == "__main__":
    main()
