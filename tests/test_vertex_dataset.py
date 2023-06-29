import numpy as np
import unittest
from dataset.vertex_grid_dataset import get_vertex_grid_dataset
from PIL import Image
from models.segformer import SegformerModel
import torch


class TestVertexGrid(unittest.TestCase):
    def test_get_vertex_grid_dataset(self):
        path = "mock_entities_data/"

        dataset = get_vertex_grid_dataset(path)
        self.assertEqual(dataset.keys(), {"test", "train", "val"})

        example = dataset['val'][1]
        example_v2 = dataset['val'][1]

        np_pixel_values = np.array(example["pixel_values"], dtype=np.uint8) * 255
        Image.fromarray(np_pixel_values[0, :, :], mode="L").show()

        np_labels = np.array(example["labels"], dtype=np.uint8) * 255
        Image.fromarray(np_labels, mode="L").show()

        # Make sure different vertices are chosen each iteration
        self.assertNotEqual(torch.sum(torch.abs(example["labels"] - example_v2["labels"])), 0)

        # Sanity check that (input + output) vertices are the same every iteration
        # Only works if pixel_values and labels have the same resolution
        # both_v1 = example["pixel_values"][0, :, :] + example["labels"]
        # both_v2 = example_v2["pixel_values"][0, :, :] + example_v2["labels"]
        # self.assertEqual(torch.sum(torch.abs(both_v1 - both_v2)), 0)

        model = SegformerModel()

        batch_size = 2
        batch = dataset['val'][:batch_size]

        loss, logits = model(**batch)

        num_classes = 2
        self.assertEqual(logits.shape, (batch_size, num_classes, 64, 64))
