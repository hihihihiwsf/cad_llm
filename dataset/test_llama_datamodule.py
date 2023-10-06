import functools
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
from datasets import load_dataset

SPECIAL_TOKENS = ["<START_Q>", "<END_Q>", "<START_A>", "<END_A>"]


def collate_fn(batch, tokenizer, max_length):
    input_sequences = [f"<START_Q>{item['question']}<END_Q>"
                       f"<START_A>{item['answer']}<END_A>" 
                       for item in batch]
    out_batch = tokenizer(
        input_sequences,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    out_batch["labels"] = out_batch["input_ids"].clone()

    return out_batch


class GSM8KDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, batch_size=32, max_length=128, data_loader_num_workers=0, local_data_dir="/home/ray/data"):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.data_loader_num_workers = data_loader_num_workers
        self.local_data_dir = local_data_dir

    def prepare_data(self):
        dataset = load_dataset("gsm8k", "main")

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(dataset['train'], [0.8, 0.2], generator=torch.Generator().manual_seed(42))
        self.test_dataset = dataset["test"]

    def setup(self, stage):
        if stage == 'fit':
            dataset = load_dataset("gsm8k", "main")
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(dataset['train'], [0.8, 0.2], generator=torch.Generator().manual_seed(42))
            self.test_dataset = dataset["test"]

    def train_dataloader(self):
        collate_partial = functools.partial(
            collate_fn,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=collate_partial, num_workers=self.data_loader_num_workers)


if __name__ == '__main__':
    
    tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens(SPECIAL_TOKENS, special_tokens=True)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    data_module = GSM8KDataModule(tokenizer=tokenizer)
    data_module.prepare_data()
    data_module.setup('fit')
    for batch in data_module.train_dataloader():
        print(batch.keys())
        break

# sentencepiece sacremoses