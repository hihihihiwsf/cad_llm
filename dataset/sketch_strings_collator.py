class SketchStringsCollator:
    def __init__(self, tokenizer, max_length=None, additional_cols=False, model_name=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.additional_cols = additional_cols
        self.model_name = model_name

    def tokenize(self, strings):
        return self.tokenizer(strings, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
    

    def llama_collate_fn(self,batch, tokenizer, max_length):
        # SPECIAL_TOKENS = ["<SYSTEM>", "<START_Q>", "<END_Q>", "<START_A>", "<END_A>"]
        # input_sequences = ["<SYSTEM> You are a cad autocomplete assistant. Q is the incomplete sketch, and A is the remaining sketch."
        #                 f"<START_Q>{item['input_text']}<END_Q>"
        #                 f"<START_A>{item['output_text']}<END_A>" 
        #                 for item in batch]
        
        input_sequences = ["<SYSTEM> You are a cad autocomplete assistant. Complete the sketch given the input sketch."
                f"{item['input_text']}"
                " <FILL_ME> "
                f"<START_A>{item['output_text']}<END_A>"
                for item in batch]
        print(input_sequences[0])
        out_batch = tokenizer(
            input_sequences,
            padding=True,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        out_batch["labels"] = out_batch["input_ids"].clone()

        return out_batch

    def __call__(self, examples):
        # Collate input_text and output_text columns
        input_text = [example['input_text'] for example in examples]
        output_text = [example['output_text'] for example in examples]
        name = [example['name'] for example in examples]

        if 'llama' in self.model_name.lower():
            return self.llama_collate_fn(examples, self.tokenizer, self.max_length)
        
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
