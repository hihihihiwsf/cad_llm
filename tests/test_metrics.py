# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

from metrics import count_accurate
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
        expected_accurate = 1
        accurate = count_accurate(labels=labels, samples=samples)
        self.assertEqual(accurate, expected_accurate)
