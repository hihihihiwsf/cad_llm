import unittest

import numpy as np
from PIL import Image

from dataset.rendered_sketch_dataset import get_rendered_sketch_dataset
from models.segformer import SegformerModel


class TestRenderedSketchesDataset(unittest.TestCase):
    def test_get_rendered_sketch_dataset(self):
        path = "mock_entities_data/"

        dataset = get_rendered_sketch_dataset(path)
        self.assertEqual(dataset.keys(), {"test", "train", "val"})

        example = dataset['val'][1]

        np_pixel_values = np.array(example["pixel_values"], dtype=np.uint8) * 255
        channel_first_pixel_values = np_pixel_values.transpose(1, 2, 0)
        rgb_np_pixel_values = channel_first_pixel_values[:, :, ::-1]
        Image.fromarray(rgb_np_pixel_values, mode="RGB").show()

        np_labels = np.array(1 - example["labels"], dtype=np.uint8) * 255
        Image.fromarray(np_labels, mode="L").show()

        model = SegformerModel("nvidia/segformer-b0-finetuned-ade-512-512")

        batch_size = 2
        batch = dataset['val'][:batch_size]

        loss, logits = model(**batch)

        num_classes = 2
        self.assertEqual(logits.shape, (batch_size, num_classes, 64, 64))
