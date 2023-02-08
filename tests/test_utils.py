from dataset.utils import get_quantized, choose_random_subset
import numpy as np
import unittest


class TestUtils(unittest.TestCase):
    def test_get_quantized(self):
        vertices = np.array([[0.5, 0.28], [-0.5, 0.28], [0.5, -0.28], [-0.5, -0.28]])
        expected_quantized = np.array([[31, 17], [-31, 17], [31, -17], [-31, -17]])

        quantized = get_quantized(vertices, n_bits=6)

        self.assertEqual(quantized.tolist(), expected_quantized.tolist())

    def test_choose_random_subset(self):
        n = 10
        subset = choose_random_subset(n, (0, 1))
        assert 1 <= len(subset) < n, f"len(subset) = {len(subset)}"

        subset = choose_random_subset(n, (0.4, 0.6))
        assert 0.4 <= len(subset) / n <= 0.6

        print("success - test_choose_random_input_output_indices")

