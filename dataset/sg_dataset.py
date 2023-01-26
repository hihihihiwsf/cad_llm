from torch.utils.data import Dataset
import numpy as np
from dataset.transforms import add_quantized, add_entities, add_subset, add_input_output


class SketchGraphsDataset(Dataset):
    def __init__(self, path, quantize_n_bits=6, subset_range=None):
        data = np.load(path, allow_pickle=True)
        self.data = [x for x in data if len(x['curves']) >= 2]
        print(f"Filtered to {len(self.data)} sketches with 2 or more curves from {len(data)} sketches")
        self.quantize_n_bits = quantize_n_bits
        self.subset_range = subset_range or [0, 1]
        assert self.subset_range[0] >= 0 and self.subset_range[1] <= 1

    def __getitem__(self, index):
        example = self.data[index]
        self._transform(example)
        return (example['input'], example['output'])

    def _transform(self, example):
        if 'entities' not in example:
            add_quantized(example, self.quantize_n_bits)
            add_entities(example)

        # Overwrite input output from previous epoch
        add_subset(example, self.subset_range)
        add_input_output(example)

    def __len__(self):
        return len(self.data)


class SketchGraphsCollator:
    def __init__(self, tokenizer, max_length=None):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, input_output_pairs):
        input_strings = [x for x, _ in input_output_pairs]
        output_strings = [y for _, y in input_output_pairs]
        tokenized_input = self.tokenizer(input_strings, padding=True, max_length=self.max_length, return_tensors='pt')
        tokenized_output = self.tokenizer(output_strings, padding=True, max_length=self.max_length, return_tensors='pt')
        batch = {
            "input_ids": tokenized_input.input_ids,
            "attention_mask": tokenized_input.attention_mask,
            "labels": tokenized_output.input_ids,
        }
        return batch
