from torch.utils.data import Dataset
import numpy as np
from dataset.transforms import to_quantized, add_entities, add_input_output


class SketchGraphsDataset(Dataset):
    def __init__(self, path, quantize_n_bits=6):
        self.data = np.load(path, allow_pickle=True)
        self.quantize_n_bits = quantize_n_bits

        if self.quantize_n_bits:
            self.data = [to_quantized(example, quantize_n_bits) for example in self.data]

        self.data = [add_entities(example) for example in self.data]

    def __getitem__(self, index):
        example = self.data[index]
        example = add_input_output(example)
        return (example['input'], example['output'])

    def __len__(self):
        return len(self.data)


class SketchGraphsCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        ret = {
            "input_ids": self.tokenizer([x for x, y in batch], padding=True, return_tensors='pt').input_ids,
            "labels": self.tokenizer([y for x, y in batch], padding=True, return_tensors='pt').input_ids,
        }
        return ret
