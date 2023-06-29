from pathlib import Path

import torch
from datasets import load_dataset

from dataset.dataset_utils import get_random_mask


def get_vertex_grid_dataset(path):
    splits = ["val", "train", "test"]
    # Match json and zipped json files
    data_files = {split: str(Path(path) / f"{split}.json*") for split in splits}

    dataset = load_dataset("json", data_files=data_files)
    dataset.set_transform(batch_split_and_render_vertices)

    return dataset


def batch_split_and_render_vertices(batch):
    batch_size = len(batch["entities"])

    # batch_input_vertices = []
    # batch_output_vertices = []

    quantize_range = 64
    k = 4  # Not configurable, determined by segformer architecture
    res = quantize_range * 4

    pixel_values = torch.zeros((batch_size, 3, res, res), dtype=torch.float)
    labels = torch.zeros((batch_size, quantize_range, quantize_range), dtype=torch.long)

    for batch_index in range(batch_size):
        entities = batch["entities"][batch_index]
        vertices = set([tuple(p) for entity in entities for p in entity])

        mask = get_random_mask(len(vertices), 0.2, 0.8)

        input_vertices = [v for i, v in enumerate(vertices) if mask[i]]
        output_vertices = [v for i, v in enumerate(vertices) if not mask[i]]

        # batch_input_vertices.append(input_vertices)
        # batch_output_vertices.append(output_vertices)

        # Expand resolution by k=4
        for x, y in input_vertices:
            pixel_values[batch_index, 0, x * k: (x + 1) * k, y * k: (y + 1) * k] = 1.

        for x, y in output_vertices:
            labels[(batch_index, x, y)] = 1

    return {
        "pixel_values": pixel_values,
        "labels": labels,
    }
