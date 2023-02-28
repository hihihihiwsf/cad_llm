from dataset.sg_dataset import SketchGraphsDataset, SketchGraphsCollator
from models.byt5 import get_byt5_model, get_new_byt5_model
from torch.utils.data import DataLoader
import torch
import time
from datetime import date
import argparse

from pathlib import Path
from experiment_log import experiment_name_to_hps
from metrics import count_accurate
from tqdm import tqdm

print('git test')

def load_model(name, checkpoint=None):
    if name == 'byt5-base':
        tokenizer, model = get_byt5_model(checkpoint)
    elif name == 'byt5-base-new':
        tokenizer, model = get_new_byt5_model()
    else:
        raise Exception(f"Unsupported model '{name}'")
    return tokenizer, model


def load_dataset(dataset_path, subset_range=None):
    train_dataset = SketchGraphsDataset(dataset_path, subset_range=subset_range)
    return train_dataset


def eval_model(model, val_dataloader, device, batch_size):
    accurate_count = 0
    total_eval_loss = 0
    # with torch.no_grad:
    for val_batch in val_dataloader:
        # Calculate eval_loss
        val_batch = {k: v.to(device) for k, v in val_batch.items()}
        val_outputs = model(**val_batch)
        total_eval_loss += float(val_outputs.loss.mean())

        # Generate samples to test accuracy (do not send labels in for generation)
        labels = val_batch["labels"]
        samples = model.generate(input_ids=val_batch["input_ids"],
                                 attention_mask=val_batch["attention_mask"],
                                 do_sample=False,
                                 max_new_tokens=labels.shape[1])

        accurate_count += count_accurate(labels=labels, samples=samples)

    val_size = len(val_dataloader) * batch_size
    accuracy = accurate_count / val_size
    eval_loss = total_eval_loss / val_size
    return {"eval_loss": eval_loss, "accuracy: ": accuracy}


def main(args):
    exp_name = args.exp_name
    exp_run_name = exp_name + '-' + date.today().strftime("%m-%d-%y")
    save_checkpoint = Path(args.chkpt) / exp_run_name

    print(f"Loading experiment '{exp_name}'")
    hps = experiment_name_to_hps[exp_name]
    print(hps)
    batch_size = hps['batch_size']
    subset_range = hps.get('subset_range')
    max_length = hps.get('max_length')
    eval_temperature = hps.get('eval_temperature', 1)

    print("Loading model...")
    start_time = time.time()
    load_checkpoint = None  # To do: add to command line args
    tokenizer, model = load_model(hps['modal'], checkpoint=load_checkpoint)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    print(f"Loading model time was {int(time.time() - start_time)} seconds")

    print("Loading data...")
    start_time = time.time()
    dataset_dir = Path(args.data)
    train_dataset = load_dataset(dataset_dir / "sg_str_train.json", subset_range=subset_range)
    val_dataset = load_dataset(dataset_dir / "sg_str_val.json", subset_range=subset_range)
    # val_dataset.data = val_dataset.data[:32]
    data_collator = SketchGraphsCollator(tokenizer=tokenizer, max_length=max_length)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=data_collator, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=data_collator, shuffle=False)

    print(f"Data loading time was {int(time.time() - start_time)} seconds")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

    num_epochs = 5
    eval_steps = 4
    save_steps = 5000
    log_steps = 500

    print("Training model...")

    num_training_steps = num_epochs * len(train_dataloader)
    progress_bar = tqdm(range(num_training_steps))
    step = 0

    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            step += 1

            if step == 1 or step % eval_steps == 0:
                print("Evaluating model...")
                model.eval()
                eval_stats = eval_model(model, val_dataloader, device, batch_size)
                print("eval_stats: ", eval_stats)
                # To do: Log eval_stats
                model.train()

            if step == 0 or step % log_steps == 0:
                pass
                # To do: Log loss, step, epoch

            if step == 1 or step % save_steps == 0:
                print("Saving checkpoint")
                model.save_pretrained(save_checkpoint)


# Example usage:
# python main.py --exp-name cad_llm_v1 --data /home/ec2-user/SageMaker/efs/data/sg_normalized --chkpt /home/ec2-user/SageMaker/efs/cad_llm_checkpoints
# python main.py --exp-name cad_llm_v1 --data ~/data/sg_normalized --chkpt ~/cad_llm_checkpoints
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, required=True,
                        help="Experiment name for lookup in experiment_log.py")
    parser.add_argument("--data", type=str, required=True,
                        help="Input folder containing the SketchGraphs preprocessed .npy files")
    parser.add_argument("--chkpt", type=str, required=True,
                        help="Output folder to save checkpoints (loading not implemented)")
    args = parser.parse_args()
    main(args)
