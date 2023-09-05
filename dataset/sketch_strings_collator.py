import numpy as np

class SketchStringsCollator:
    def __init__(self, tokenizer, max_length=None, additional_cols=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.additional_cols = additional_cols

    def tokenize(self, strings):
        return self.tokenizer(strings, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")

    def __call__(self, examples):
        # Collate input_text and output_text columns
        input_text = [example['input_text'] for example in examples]
        output_text = [example['output_text'] for example in examples]
        name = [example['name'] for example in examples]
        
        

        
        input_text_revised, output_text_revised = [], []
        for example in examples:
            single_ents = []
            for i, x in enumerate(example['input_text'].split(";")):
                if x:
                    x_token = self.tokenizer.encode(x)[1:-1]
                    if len(x_token) == 15:
                        x = self.tokenizer.batch_decode(x_token)
                        x_point = str(int(np.mean([int(x[0]), int(x[4]), int(x[8]), int(x[12])])))
                        y_point = str(int(np.mean([int(x[2]), int(x[6]), int(x[10]), int(x[14])])))
                        radius = str(int(abs(int(x[10]) - int(x[6]))))
                        single_ents.append(x_point+","+y_point+","+radius+";")
                    else:
                        single_ents.append(x+";")
        
            in_txt = ''.join(single_ents)
            input_text_revised.append(in_txt)
        
        for example in examples:
            single_ents = []
            for i, x in enumerate(example['output_text'].split(";")):
                if x:
                    x_token = self.tokenizer.encode(x)[1:-1]
                    if len(x_token) == 15:
                        x = self.tokenizer.batch_decode(x_token)
                        x_point = str(int(np.mean([int(x[0]), int(x[4]), int(x[8]), int(x[12])])))
                        y_point = str(int(np.mean([int(x[2]), int(x[6]), int(x[10]), int(x[14])])))
                        radius = str(int(abs(int(x[10]) - int(x[6]))))
                        single_ents.append(x_point+","+y_point+","+radius+";")
                    else:
                        single_ents.append(x+";")
        
            out_text = ''.join(single_ents)
            output_text_revised.append(out_text)

        input_text = input_text_revised
        output_text = output_text_revised
        
        # Encode input and output
        tokenized_input = self.tokenize(input_text)
        tokenized_output = self.tokenize(output_text)
        labels = tokenized_output.input_ids
        # replace padding token id's of the labels by ignore_index=-100 so it's ignored by the loss
        labels[labels == self.tokenizer.pad_token_id] = -100

        batch = {
            "input_ids": tokenized_input.input_ids,
            "attention_mask": tokenized_input.attention_mask,
            "labels": labels,
            "input_text": input_text,
            "output_text": output_text,
            "name": name,
        }

        if self.additional_cols:
            for col in self.additional_cols:
                batch[col] = [example[col] for example in examples]

        return batch
