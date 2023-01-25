from dataset.base_dataset import SketchGraphsDataset, SketchGraphsCollator
from models.byt5 import get_byt5_model, get_new_byt5_model
from transformers import Seq2SeqTrainer, TrainingArguments
import time
from datetime import date
import argparse
import torch
from pathlib import Path
from experiment_log import experiment_name_to_hps


def load_model(name):
    if name == 'byt5-base':
        tokenizer, model = get_byt5_model()
    elif name == 'byt5-base-new':
        tokenizer, model = get_new_byt5_model()
    else:
        raise Exception(f"Unsupported model '{name}'")
    return tokenizer, model


def load_dataset(dataset_path):
    train_dataset = SketchGraphsDataset(dataset_path)
    return train_dataset


def main(args):
    save_checkpoint = Path(args.chkpt) / date.today().strftime("%m-%d-%y")

    print(f"Loading experiment '{args.exp_name}'")
    hps = experiment_name_to_hps[args.exp_name]
    print(hps)

    print("Loading model...")
    start_time = time.time()
    tokenizer, model = load_model(hps['modal'])
    if torch.cuda.is_available():
        model.to('cuda')
    print(f"Loading model time was {int(time.time() - start_time)} seconds")

    print("Loading data...")
    start_time = time.time()
    dataset_dir = Path(args.data)
    train_dataset = load_dataset(dataset_dir / "sg_obj_train.npy")
    data_collator = SketchGraphsCollator(tokenizer)
    print(f"Data loading time was {int(time.time() - start_time)} seconds")

    training_args = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        # save_steps=100,
        save_steps=50000,
        num_train_epochs=5,
        output_dir=save_checkpoint,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
        compute_metrics=None,
        data_collator=data_collator
    )

    print("Training model...")
    trainer.train()


############ Evaluate ############

# generate n samples with t temperature
# def generate_samples_with_temp(txt, n_samples, temp):
#     to_tokenizer = [txt for i in range(n_samples)]
#     outputs = model.generate(tokenizer(to_tokenizer, return_tensors='pt', padding=True).input_ids.to('cuda'),
#                              do_sample=True, max_length=128, temperature=temp)
#     # right answer is 1:ggg 2:rr 3:pp 4:gg 5:r 6:pp 7:pp
#     results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
#     return results


# Example usage:
# python main.py --data /home/ec2-user/SageMaker/efs/data/sg_normalized --chkpt /home/ec2-user/SageMaker/efs/cad_llm_checkpoints

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
