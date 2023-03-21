import numpy as np
import unittest
from preprocess.preprocessing import preprocess_sketch

sketch_obj_dict = {
    'name': 'val_00043061',
    'vertices': np.array([
        [0.5,  0.28],
        [-0.5,  0.28],
        [0.5, -0.28],
        [-0.5, -0.28]
    ]),
    'curves': np.array([
        [1, 2],
        [3, 4],
        [1, 3],
        [2, 4]
    ]),
}

expected_entity_strings = ['-31,-17,-31,17;', '-31,-17,31,-17;', '-31,17,31,17;', '31,-17,31,17;']
expected_new_tokens_entity_strings = ['<-31><-17><-31><17>;', '<-31><-17><31><-17>;',
                                      '<-31><17><31><17>;', '<31><-17><31><17>;']
expected_user_ordered_entity_strings = ['<31><17><-31><17>;', '<31><-17><-31><-17>;',
                                        '<31><17><31><-17>;', '<-31><17><-31><-17>;']


class TestSketchLLM(unittest.TestCase):
    def test_generate_random_input_output(self):
        sketch_str_dict = preprocess_sketch(sketch_obj_dict, quantize_bits=6)
        self.assertEqual(sketch_str_dict["entities"], expected_entity_strings)

        sketch_str_dict = preprocess_sketch(sketch_obj_dict, quantize_bits=6, new_tokens=True)
        self.assertEqual(sketch_str_dict["entities"], expected_new_tokens_entity_strings)
        self.assertEqual(sketch_str_dict["user_ordered_entities"], expected_user_ordered_entity_strings)
