# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

from metrics import calculate_accuracy
import torch
import unittest


class TestMetrics(unittest.TestCase):
    def test_get_quantized(self):
        labels = torch.tensor([
            [48, 54, 52, 47, 48, 52, 52, 47, 54, 52, 47, 48, 52, 52, 62, 1, -100, -100, -100, -100, -100, -100],
            [48, 54, 52, 47, 52, 47, 54, 52, 47, 48, 52, 62, 1, -100, -100, -100, -100, -100, -100, -100, -100, -100]
        ])
        samples = torch.tensor([
            [0, 48, 54, 52, 47, 48, 52, 52, 47, 54, 52, 47, 48, 52, 52, 62, 1, 0, 0, 0],
            [0, 48, 54, 52, 47, 48, 52, 47, 48, 54, 52, 47, 52, 62, 1, 0, 0, 0, 0, 0],
        ])
        expected_accuracy = 0.5
        accuracy = calculate_accuracy(labels=labels, samples=samples)
        self.assertEqual(accuracy, expected_accuracy)
