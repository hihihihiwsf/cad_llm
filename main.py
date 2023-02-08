from dataset.sg_dataset import SketchGraphsDataset, SketchGraphsCollator
from models.byt5 import get_byt5_model, get_new_byt5_model
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import time
from datetime import date
import argparse
import torch
from pathlib import Path
from experiment_log import experiment_name_to_hps
from metrics import get_compute_metrics
import comet_ml


def load_model(name):
    if name == 'byt5-base':
        tokenizer, model = get_byt5_model()
    elif name == 'byt5-base-new':
        tokenizer, model = get_new_byt5_model()
    else:
        raise Exception(f"Unsupported model '{name}'")
    return tokenizer, model


def load_dataset(dataset_path, subset_range=None):
    train_dataset = SketchGraphsDataset(dataset_path, subset_range=subset_range)
    return train_dataset


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
    tokenizer, model = load_model(hps['modal'])
    if torch.cuda.is_available():
        model.to('cuda')
    print(f"Loading model time was {int(time.time() - start_time)} seconds")

    print("Loading data...")
    start_time = time.time()
    dataset_dir = Path(args.data)
    train_dataset = load_dataset(dataset_dir / "sg_obj_train.npy", subset_range=subset_range)
    val_dataset = load_dataset(dataset_dir / "sg_obj_val.npy", subset_range=subset_range)
    data_collator = SketchGraphsCollator(tokenizer=tokenizer, max_length=max_length)
    print(f"Data loading time was {int(time.time() - start_time)} seconds")

    comet_ml.init(project_name="cad-llm")

    compute_metrics = get_compute_metrics(
        exp_name=exp_run_name,
        dataset=val_dataset,
        tokenizer=tokenizer,
        temperature=eval_temperature,
    )

    training_args = Seq2SeqTrainingArguments(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=1,
        num_train_epochs=5,
        save_total_limit=2,
        save_steps=5000,
        output_dir=str(save_checkpoint),
        logging_first_step=True,
        evaluation_strategy="steps",
        # eval_steps=1000,
        eval_steps=1,
        generation_max_length=max_length,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
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
