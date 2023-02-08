from torch.utils.data import Dataset
import numpy as np
from dataset.sketch_llm import SketchLLM


class SketchGraphsDataset(Dataset):
    def __init__(self, path, quantize_n_bits=6, subset_range=None):
        data = np.load(path, allow_pickle=True)
        self.data = [SketchLLM(sketch_dict, quantize_n_bits=quantize_n_bits) for sketch_dict in data]
        self.data = [sketch for sketch in self.data if len(sketch.curves) >= 2]
        print(f"Filtered to {len(self.data)} sketches with 2 or more curves from {len(data)} sketches")

        self.subset_range = subset_range or [0, 1]
        assert self.subset_range[0] >= 0 and self.subset_range[1] <= 1

    def __getitem__(self, index):
        """
        Generates input/output by selecting a subset of entities
        Note: Overrides subset selection info from previous calls

        Returns (input_text, output_text)
        """
        return self.data[index].generate_random_input_output(subset_range=self.subset_range)

    def get_sketch(self, index):
        """
        Returns the SketchLLM object at index
        """
        return self.data[index]

    def __len__(self):
        return len(self.data)


class SketchGraphsCollator:
    def __init__(self, tokenizer, max_length=None):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def tokenize(self, strings):
        return self.tokenizer(strings, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt')

    def __call__(self, input_output_pairs):
        input_strings = [x for x, _ in input_output_pairs]
        output_strings = [y for _, y in input_output_pairs]

        tokenized_input = self.tokenize(input_strings)
        tokenized_output = self.tokenize(output_strings)

        labels = tokenized_output.input_ids
        # replace padding token id's of the labels by ignore_index=-100 so it's ignored by the loss
        labels[labels == self.tokenizer.pad_token_id] = -100

        batch = {
            "input_ids": tokenized_input.input_ids,
            "attention_mask": tokenized_input.attention_mask,
            "labels": labels,
        }
        return batch
