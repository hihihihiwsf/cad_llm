class SketchStringsCollator:
    def __init__(self, tokenizer, max_length=None):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def tokenize(self, strings):
        return self.tokenizer(strings, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")

    def __call__(self, examples):
        # Collate input_text and output_text columns
        input_text = [example['input_text'] for example in examples]
        output_text = [example['output_text'] for example in examples]

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
        }

        return batch
