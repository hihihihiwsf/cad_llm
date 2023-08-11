import torch

class SketchStringsCollator:
    def __init__(self, tokenizer, max_length=None):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def tokenize(self, strings):
        return self.tokenizer(strings, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")

    def __call__(self, examples):
        # Collate input_text and output_text columns
        input_text = [example['input_text'] for example in examples]
        output_text = [example['output_text'] for example in examples]

        input_entities, single_ents_length = [], []
        output_entities, single_ents_length_out = [], []
        for example in examples:
            single_ents = ["C"+x+";" for x in example['input_text'].split(";") if x][:30]
            input_entities.extend(single_ents)
            single_ents_length.append(len(single_ents))

            single_ents_out = ["C"+x+";" for x in example['output_text'].split(";") if x][:29]
            single_ents_out.extend(["C" + "END"*15])
            output_entities.extend(single_ents_out)
            single_ents_length_out.append(len(single_ents_out))
            
            
        # Encode input and output
        tokenized_input = self.tokenize(input_text)
        tokenized_output = self.tokenize(output_text)
        labels = tokenized_output.input_ids
        tokenized_single_entity = self.tokenize(input_entities)
        tokenized_single_entity_output = self.tokenize(output_entities)

        
        # replace padding token id's of the labels by ignore_index=-100 so it's ignored by the loss
        labels[labels == self.tokenizer.pad_token_id] = -100
        batch_att_mask = torch.zeros(labels.shape[0], max(single_ents_length))
        for i, j in enumerate(single_ents_length):
            batch_att_mask[i, :j] = 1
        
        batch = {
            "input_ids": tokenized_input.input_ids,
            "attention_mask": tokenized_input.attention_mask,
            "labels": labels,
            "input_text": input_text,
            "output_text": output_text,
            "input_entities": tokenized_single_entity,
            "input_ent_length": single_ents_length,
            "batch_att_mask": batch_att_mask,
            "output_entities": tokenized_single_entity_output,
            "output_ent_length": single_ents_length_out,
            
        }

        return batch
