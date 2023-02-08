from dataset.utils import choose_random_subset
from dataset.sketch_llm import SketchLLM
import numpy as np

import unittest

sketch_dict = {
    'name': 'val_00043061',
    'vertices': np.array([
        [0.5,  0.28],
        [-0.5,  0.28],
        [0.5, -0.28],
        [-0.5, -0.28]
    ]),
    'curves': np.array([
        [1, 2, 0, 0],
        [3, 4, 0, 0],
        [1, 3, 0, 0],
        [2, 4, 0, 0]
    ]),
}

expected_entity_strings = ['-31,-17,-31,17;', '-31,-17,31,-17;', '-31,17,31,17;', '31,-17,31,17;']


class TestSketchLLM(unittest.TestCase):
    def test_generate_random_input_output(self):
        sketch = SketchLLM(sketch_dict, quantize_n_bits=6)
        input_text, output_text = sketch.generate_random_input_output(subset_range=[0, 1])

        all_sorted_entity_strings = [ent.to_string() for ent in sketch.entities]
        self.assertEqual(all_sorted_entity_strings, expected_entity_strings)

        input_entity_strings = set([s + ";" for s in input_text.split(";") if s])
        completion_entities = sketch.get_completion_strings()

        self.assertEqual(input_entity_strings.union(completion_entities), set(all_sorted_entity_strings))
        self.assertEqual(input_entity_strings.intersection(completion_entities), set())
        self.assertTrue(output_text in completion_entities)
