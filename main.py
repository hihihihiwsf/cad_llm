"""
Train a CAD LLM model
"""


from dataset.sg_dataset import SketchGraphsDataset, SketchGraphsCollator
from models.byt5 import ByT5Model
from torch.utils.data import DataLoader

from experiment_log import experiment_name_to_hps
import multiprocessing
from args.train_args import get_training_args

from pathlib import Path

try:
    import comet_ml  # Import before torch
except ImportError:
    pass
import torch
from pytorch_lightning import Trainer



def get_trainer(args, loggers, callbacks=None):
    """Get the PyTorch Lightning Trainer"""
    log_every_n_steps = 100

    # loggers = [pl.loggers.CSVLogger(".", name="log")]
    trainer = Trainer(
        callbacks=callbacks,
        accelerator="cpu", #args.gpus,  # cpu / gpu
        devices=1,
        # logger=loggers,
        max_epochs=1,
        log_every_n_steps=log_every_n_steps,
        resume_from_checkpoint=None,
    )

    return trainer


def main():
    """Entry point for our training script"""

    args = get_training_args()

    print(f"Loading experiment '{args.exp_name}'")
    hps = experiment_name_to_hps[args.exp_name]
    print(hps)
    batch_size = hps['batch_size']
    subset_range = hps.get('subset_range')
    max_length = hps.get('max_length')
    num_cpus = multiprocessing.cpu_count() if args.parallel else 1

    save_checkpoint = Path(args.chkpt) / args.exp_name

    print("Loading model...")
    load_checkpoint = None  # To do: add to command line args
    no_pretrain = hps.get("no_pretrain", False)
    model_name = hps['modal']
    model = ByT5Model(model_name=model_name, checkpoint=load_checkpoint, no_pretrain=no_pretrain)

    print("Loading data...")
    dataset_dir = Path(args.dataset)

    train_dataset = SketchGraphsDataset(dataset_dir=dataset_dir, split="train", subset_range=subset_range)
    val_dataset = SketchGraphsDataset(dataset_dir=dataset_dir, split="val", subset_range=subset_range)

    data_collator = SketchGraphsCollator(tokenizer=model.tokenizer, max_length=max_length)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=data_collator, shuffle=True,
                                  num_workers=num_cpus)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=data_collator, shuffle=False,
                                num_workers=num_cpus)

    # # We save the logs to the experiment directory
    # loggers = util.get_loggers(args, exp_name_dir, args_file)
    # # Save the args to the checkpoint directory to use to resume
    # with open(args_file, "w", encoding="utf8") as f:
    #     json.dump(vars(args), f, indent=4)

    trainer = get_trainer(args, loggers=None, callbacks=None)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    #
    # exp_dir = Path(args.exp_dir)
    # exp_name_dir = exp_dir / args.exp_name
    # if not exp_name_dir.exists():
    #     exp_name_dir.mkdir(parents=True)
    # if not exp_name_dir.exists():
    #     exp_name_dir.mkdir(parents=True)
    # checkpoint_dir = Path(args.checkpoint_dir)
    # checkpoint_exp_name_dir = checkpoint_dir / args.exp_name
    # if not checkpoint_exp_name_dir.exists():
    #     checkpoint_exp_name_dir.mkdir(parents=True)
    #
    # args_file = checkpoint_exp_name_dir / "args.json"
    # # If we are using Sagemaker, check the checkpoints directory
    #
    #
    # # TRAINING
    # train_dataset = load_dataset(args, split="train")
    # val_dataset = load_dataset(args, split="val")
    #
    # train_model(args, exp_name_dir, checkpoint_exp_name_dir, loggers, train_dataset, val_dataset)
    #


# Example usage:
# python main.py --exp_name cad_llm_test --dataset ~/data/sg_normalized --chkpt ~/cad_llm_checkpoints
if __name__ == "__main__":
    main()
