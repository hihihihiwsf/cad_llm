class SketchStringsCollator:
    def __init__(self, tokenizer, max_length=None, additional_cols=False, model_name=None):
        
        self.max_length = max_length
        self.additional_cols = additional_cols
        self.model_name = model_name
        SPECIAL_TOKENS = ["<SYSTEM>", "<START_Q>", "<END_Q>", "<START_A>", "<END_A>"]
        self.system_prompt = """You are a CAD sketch autocomplete assistant. A drawing of a CAD sketch will be given to you and you try to complete it in a meaningful way.
        \nThe drawing is represented by 3 types of entities: lines, curves, and circles.\n\nEach point is quantized to 6 bits, i.e. from 1 to 64 and it's shown with its x, y coordinates respectively.
        A point example: x_1,y_1\nx_i is the x coordinate and y_i is the y coordinate of a point.\n\nA line is represented by a sequence of 2 points.\nA line example: x_1,y_1,x_2,y_2;\n\n
        A curve is represented by a sequence of 3 points.\nA curve example: x_1,y_1,x_2,y_2,x_3,y_3;\n\nA circle is represented by a sequence of 4 points.\nA circle example: x_1,y_1,x_2,y_2,x_3,y_3,x_4,y_4"""
        self.tokenizer = tokenizer

    def tokenize(self, strings):
        return self.tokenizer(strings, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
    

    def llama_collate_fn(self,batch, tokenizer, max_length):
        # "<SYSTEM> You are a cad autocomplete assistant. Q is the incomplete sketch, and A is the remaining sketch."
        # " '''ONLY OUTPUT THE ANSWER. DO NOT REPEAT THE QUESTION.''' "
        input_sequences = [
                        f"<START_Q>{item['input_text']}<END_Q>"
                        f"<START_A>{item['output_text']}<END_A>" 
                        for item in batch]

        prefix_sequences = [
                f"<START_Q>{item['input_text']}<END_Q>"
                f"<START_A>" 
                for item in batch]
        

        # input_sequences = ["<SYSTEM> You are a cad autocomplete assistant. Complete the sketch given the input sketch."
        #         f"{item['input_text']}"
        #         " <FILL_ME> "
        #         f"<START_A>{item['output_text']}<END_A>"
        #         for item in batch]
        # print(input_sequences[0])
        out_batch = tokenizer(
            input_sequences,
            padding=True,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        
        out_batch["labels"] = out_batch["input_ids"].clone()
        
        generation_batch = tokenizer(
            prefix_sequences,
            padding=True,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        
        res = {
            "input_ids": out_batch["input_ids"],
            "attention_mask": out_batch["attention_mask"],
            "labels": out_batch["input_ids"],
            "generation_input_ids": generation_batch['input_ids'],
            "generation_attention_mask": generation_batch['attention_mask'],
        }
        

        return res

    def __call__(self, examples):
        # Collate input_text and output_text columns
        input_text = [example['input_text'] for example in examples]
        output_text = [example['output_text'] for example in examples]
        name = [example['name'] for example in examples]

        if 'llama' in self.model_name.lower():
            batch = self.llama_collate_fn(examples, self.tokenizer, self.max_length)
            batch["input_text"] = input_text
            batch["output_text"] = output_text
            batch["name"] = name
            # return batch
        
        else:
            # Encode input and output
            tokenized_input = self.tokenize(input_text)

            tokenized_output = self.tokenize(output_text)
            labels = tokenized_output.input_ids
            # replace padding token id's of the labels by ignore_index=-100 so it's ignored by the loss
            # labels[labels == self.tokenizer.pad_token_id] = -100

            batch = {
                "input_ids": tokenized_input.input_ids,
                "attention_mask": tokenized_input.attention_mask,
                "labels": tokenized_input.input_ids,
                "input_text": input_text,
                "output_text": output_text,
                "name": name,
            }

        if self.additional_cols:
            for col in self.additional_cols:
                batch[col] = [example[col] for example in examples]

        return batch
