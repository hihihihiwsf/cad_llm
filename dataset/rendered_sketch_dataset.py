from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from datasets import load_dataset

from dataset.dataset_utils import get_random_mask
from geometry.parse import get_curves


def get_rendered_sketch_dataset(path):
    splits = ["val", "train", "test"]
    data_files = {split: str(Path(path) / f"{split}.json.zip") for split in splits}

    dataset = load_dataset("json", data_files=data_files)
    dataset.set_transform(batch_split_and_render)

    return dataset


def batch_split_and_render(batch, input_res=256, output_res=64):
    """
    Assumes entities are quantized to range 0, ..., 63
    """
    batch_size = len(batch["entities"])

    batch_input_entities = []
    batch_output_entities = []

    pixel_values = torch.zeros((batch_size, 3, input_res, input_res), dtype=torch.float)
    labels = torch.zeros((batch_size, output_res, output_res), dtype=torch.long)

    for batch_index in range(batch_size):
        entities = batch["entities"][batch_index]

        # Split entities to input and output
        mask = get_random_mask(len(entities), 0.2, 0.8)
        input_entities = [v for i, v in enumerate(entities) if mask[i]]
        output_entities = [v for i, v in enumerate(entities) if not mask[i]]
        # batch_input_entities.append(input_entities)
        # batch_output_entities.append(output_entities)

        # Render input and output curves
        rendered_input = render_sketch(input_entities, pixel_size=input_res)
        rendered_output = render_sketch(output_entities, pixel_size=output_res)

        # Set pixel_values to scaled rendered_input
        pixel_values[batch_index, :, :, :] = torch.tensor(rendered_input, dtype=torch.float) / 255

        # Set labels to 1 if the rendered_output is not white, 0 otherwise
        labels[batch_index, :, :] = torch.tensor(np.any((rendered_output != 255), axis=0))

    return {
        "pixel_values": pixel_values,
        "labels": labels,
    }


def render_sketch(entities, pixel_size, linewidth=2):
    # Shift coordinates by 0.5 for symmetric render
    entities = [[[x + 0.5, y + 0.5] for x, y in entity] for entity in entities]

    curves = get_curves(entities)
    assert all(curve for curve in curves)

    dpi = 100
    fig, ax = plt.subplots(figsize=(pixel_size / dpi, pixel_size / dpi), dpi=dpi)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    ax.set_axis_off()
    ax.set_xlim(left=0, right=64)
    ax.set_ylim(bottom=0, top=64)

    for curve in curves:
        curve.draw(ax=ax, linewidth=linewidth)

    fig.canvas.draw()
    pil_image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    plt.close(fig)

    np_image = np.asarray(pil_image.convert('RGB'))
    np_image = np_image.transpose((2, 0, 1))

    return np_image
