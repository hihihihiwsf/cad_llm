from dataset.base_dataset import SketchGraphsDataset, SketchGraphsCollator
from models.byt5 import get_byt5_model
from transformers import Seq2SeqTrainer, TrainingArguments
import time
from datetime import date
import argparse
from pathlib import Path


def main(dataset_path, save_checkpt, load_checkpt=None):
    start_time = time.time()
    print("Loading model...")
    tokenizer, model = get_byt5_model(checkpoint=load_checkpt)
    model_loaded_time = time.time()
    print(f"Loading model time was {int(model_loaded_time - start_time)} sec")

    print("Loading data...")
    train_dataset = SketchGraphsDataset(dataset_path / "sg_obj_train.npy")
    data_collator = SketchGraphsCollator(tokenizer)
    data_loaded_time = time.time()
    print(f"Data loading time was {int(data_loaded_time - model_loaded_time)} sec")

    training_args = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        # save_steps=100,
        save_steps=50000,
        num_train_epochs=5,
        output_dir=save_checkpt,
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
    parser.add_argument("--data", type=str, required=True,
                        help="Input folder containing the SketchGraphs preprocessed .npy files")
    parser.add_argument("--chkpt", type=str, required=True,
                        help="Output folder to save checkpoints (loading not implemented)")
    args = parser.parse_args()

    dataset_path = Path(args.data)
    save_checkpoint = Path(args.chkpt) / date.today().strftime("%m-%d-%y")
    main(dataset_path=dataset_path, save_checkpt=save_checkpoint, load_checkpt=None)
