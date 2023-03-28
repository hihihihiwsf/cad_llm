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
import copy
from transformers import AutoTokenizer, T5TokenizerFast
from torch.utils.data import Dataset, DataLoader
import json


class SketchGraphsDataset(Dataset):
    def __init__(self, args, split):
        path = Path(args.dataset) / f"sg_str_{split}.json"
        with open(path, "r") as f:
            self.data = json.load(f)

        self.order = args.order
        assert self.order in ["sorted", "user", "random"]
        self.entities_col = "user_ordered_entities" if self.order == "user" else "entities"
        self.min_input_percent = args.min_input_percent
        self.max_input_percent = args.max_input_percent
        assert self.min_input_percent >= 0 and self.max_input_percent <= 1

    def __getitem__(self, index):
        """
        Applies a random mask to the entities of sketch
        Returns (input_text, output_text)
        """
        sketch_dict = self.data[index]
        entities = sketch_dict[self.entities_col]

        return "".join(entities)

    def __len__(self):
        return len(self.data)


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

    # train_args = copy.deepcopy(args)
    # val_args = copy.deepcopy(args)

    # train_args.min_input_percent, train_args.max_input_percent = 0.15, 0.17
    # val_args.min_input_percent, val_args.max_input_percent = 0., 1.

    # train_dataloader = get_sketchgraphs_dataloader(tokenizer=model.tokenizer, args=train_args, split="train", shuffle=True)
    # val_dataloader = get_sketchgraphs_dataloader(tokenizer=model.tokenizer, args=val_args, split="val", shuffle=False)

    # old_tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # tokenizer = old_tokenizer.train_new_from_iterator(train_dataloader, 70)

    ds = SketchGraphsDataset(args, "train")
    iterator = DataLoader(ds, batch_size=64)
    old_tokenizer = AutoTokenizer.from_pretrained("t5-base")
    new_tokenizer = old_tokenizer.train_new_from_iterator(text_iterator=iterator, vocab_size=100)

    new_tokenizer.save_pretrained("./t5-base-vocab-100")
    print('stop')

if __name__ == "__main__":
    main()
