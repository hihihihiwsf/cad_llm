from torch.utils.data import Dataset
import numpy as np
from dataset.utils import get_quantized, choose_random_io_indices
from dataset.entity import Entity

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
            self._add_entities(example)

        # Overwrite input output from previous epoch
        self._add_random_input_output(example)

    def _add_entities(self, example):
        """
        Add 'entities' - a list of lists combining the information in vertices and curves
        """
        vertices = example['vertices']
        if self.quantize_n_bits:
            vertices = get_quantized(vertices, self.quantize_n_bits)
        point_lists = [[tuple(vertices[i - 1]) for i in c if i] for c in example['curves']]

        example['entities'] = [Entity(points=points) for points in point_lists]
        return example

    def _add_random_input_output(self, example):
        entities = example['entities']
        rand_indices = choose_random_io_indices(len(entities), subset_range=self.subset_range)

        example['subset_entities'] = sorted([entities[i] for i in rand_indices['subset']], key=lambda ent: ent.points)

        completion_entities = [entities[i] for i in rand_indices['completion']]
        example['completion_entities'] = sorted(completion_entities, key=lambda ent: ent.points)
        example['output_entities'] = sorted(completion_entities, key=lambda ent: ent.points)

        example['input'] = Entity.entities_to_string(example['subset_entities'])
        example['output'] = Entity.entities_to_string(example['output_entities'])
        return example

    def get_completions(self, index):
        example = self.data[index]
        return {ent.to_string() for ent in example['completion_entities']}

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
