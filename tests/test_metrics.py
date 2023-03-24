# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

from metrics import calculate_accuracy, calculate_first_ent_accuracy
import torch
import unittest


class TestMetrics(unittest.TestCase):
    def test_calculate_accuracy(self):
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

    def test_calculate_first_ent_accuracy(self):
        labels = ["<1><2><3><4>;<5><6><7><8>;"]
        samples = ["<5><6><7><8>;"]
        expected_accuracy = 1

        accuracy = calculate_first_ent_accuracy(string_labels=labels, string_samples=samples)
        self.assertEqual(accuracy, expected_accuracy)
