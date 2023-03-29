# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

from metrics import calculate_accuracy, calculate_first_ent_accuracy
import torch
import unittest
from geometry.parse import get_point_entities


labels = [
    "<1><2><3><4>;<5><6><7><8>;",
    "<1><2><3><4>;<5><6><7><8>;",
    "<1><2><3><4>;<5><6><7><8>;",
    "<1><2><3><4>;<5><6><7><8>;",
    "<1><2><3><4>;<5><6><7><8>;",
]
samples = [
    "<5><6><7><8>;",
    "<7><8><5><6>;",
    "<7><8><5><6><9>;",  # invalid
    "<3><4><1><2>;<7><8><5><6>;",
    "<3><1><4><2>;<7><8><5><6>;",  # bad
]


class TestMetrics(unittest.TestCase):
    def test_calculate_accuracy(self):
        point_labels = [get_point_entities(label) for label in labels]
        point_samples = [get_point_entities(sample) for sample in samples]

        expected_accuracy = 1 / 5
        accuracy = calculate_accuracy(labels=point_labels, samples=point_samples)
        self.assertEqual(accuracy, expected_accuracy)

    def test_calculate_first_ent_accuracy(self):
        point_labels = [get_point_entities(label) for label in labels]
        point_samples = [get_point_entities(sample) for sample in samples]

        expected_accuracy = 3 / 5
        accuracy = calculate_first_ent_accuracy(labels=point_labels, samples=point_samples)
        self.assertEqual(accuracy, expected_accuracy)
