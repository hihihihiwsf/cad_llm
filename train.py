from transformers import Seq2SeqTrainer, T5ForConditionalGeneration, AutoTokenizer, TrainingArguments
import json
import random 
from torch.utils.data import Dataset, DataLoader

# NOTEBOOK
# !pip install transformers datasets
# from train import run_model
# model_name = "ByT5"
# preprocessed_dataset = "/home/ec2-user/SageMaker/efs/cad_llm_evan/data_train.json"
# save_checkpt = '/home/ec2-user/SageMaker/efs/cad_llm_checkpoints/'
# run_model(model_name=model_name, dataset_path=preprocessed_dataset, save_checkpt=save_checkpt, load_checkpt=None)

############ Data loader ############

def repr_input(input_set):
    # sort the input set so we have canonical ordering of entities
    input_set = sorted(input_set)
    ret = ''
    for ent in input_set:
        to_add = repr(ent).replace('[','').replace(']','').replace(' ','')
        ret+= to_add + ';'
    return ret
    
def entry_to_io(entry):
    entry = entry['entities']
    # randomly choose a size 1 to the length of entry
    size = random.randint(1,len(entry))
    # select a subset of size from the entry, without replacement
    subset = random.sample(entry,size)
    subset_size = len(subset)
    # split the subset into evertyhing and one last thing
    input_set = subset[:subset_size-1]
    output_set = subset[subset_size-1]
    input_str = repr_input(input_set)
    output_str = repr_input([output_set])
    return (input_str, output_str)


class FactoringDataset(Dataset):
    def __init__(self, dataset_itself):
        self.data = list()
        self.data = dataset_itself
        
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)


def get_dataset(path):
    print("Loading data from path:", path)
    with open(path) as f:
        json_data = json.load(f)
    
    data = [entry_to_io(x) for x in json_data]

    dataset = FactoringDataset(data)

    return dataset

class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        ret = {"input_ids": self.tokenizer([entry[0] for entry in batch], padding=True, return_tensors='pt').input_ids, 
                "labels": self.tokenizer([entry[1] for entry in batch], padding=True, return_tensors='pt').input_ids}
        return ret
    

############ Model ############

def get_model(name, checkpt=None):
    if name == "ByT5":
        tokenizer = AutoTokenizer.from_pretrained('google/byt5-base')
        checkpt = checkpt or 'google/byt5-base'
        model = T5ForConditionalGeneration.from_pretrained(checkpt)
        model.to('cuda')
        return tokenizer, model
    
    raise Exception(f"model '{name}' not found")


############ Evaluate ############

# generate n samples with t temperature
def generate_samples_with_temp(txt, n_samples, temp):
    to_tokenizer = [txt for i in range(n_samples)]
    outputs = model.generate(tokenizer(to_tokenizer, return_tensors='pt', padding=True).input_ids.to('cuda'), do_sample=True, max_length=128, temperature = temp)
    # right answer is 1:ggg 2:rr 3:pp 4:gg 5:r 6:pp 7:pp
    results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return results

# generate_samples_with_temp('-31,5,-31,7;-31,7,30,7;29,-7,29,5;29,-7,31,-7;29,-4,29,-7;29,-2,29,0;31,6,31,-7;31,6,31,7,30,7;', 10, 1.0)


############ Run ############

def run_model(model_name, dataset_path, save_checkpt, load_checkpt=None):
    print("Loading model...")
    tokenizer, model = get_model(model_name, checkpt=load_checkpt)

    print("Loading data...")
    dataset = get_dataset(dataset_path)
    data_collator = Collator(tokenizer)
    
    # save_checkpt = add timestamp?

    training_args = TrainingArguments(
        per_device_train_batch_size=2, 
        gradient_accumulation_steps=1,
        save_steps = 76100*2,
        num_train_epochs=1,
        output_dir = save_checkpt,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args = training_args,
        train_dataset=dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
        compute_metrics=None,
        data_collator=data_collator
    )

    print("Training model...")    
    trainer.train()
 