import numpy as np
import unittest
from preprocess import convert_deepmind_to_obj


dm_sketch = {'entitySequence': {'entities': [
    {'lineEntity': {'start': {'x': 0.16666649999141783,
      'y': 0.1666664999914178},
     'end': {'x': -0.16666649999141775, 'y': 0.16666649999141786}}},
   {'lineEntity': {'start': {'x': 0.16666649999141772,
      'y': -0.16666649999141772},
     'end': {'x': -0.16666649999141786, 'y': -0.1666664999914177}}},
   {'lineEntity': {'isConstruction': True,
     'start': {'x': 0.1666664999914178, 'y': 0.16666649999141783},
     'end': {'x': 0.16666649999141767, 'y': -0.16666649999141772}}},
   {'lineEntity': {'isConstruction': True,
     'start': {'x': -0.16666649999141775, 'y': 0.16666649999141786},
     'end': {'x': -0.1666664999914179, 'y': -0.16666649999141767}}},
   {'pointEntity': {'point': {'x': 7.238184345098862e-18,
      'y': 3.626574587098518e-17}}},
   {'circleArcEntity': {'center': {'x': -0.16666649999141783},  # removed almost 0 y val for test
     'arcParams': {'start': {'x': -0.16666649999141783,
       'y': 0.16666649999141783},
      'end': {'x': -0.16666649999141786, 'y': -0.16666649999141767}}}},
   {'circleArcEntity': {'center': {'x': 0.16666649999141775,
      'y': 3.942260482986438e-17},
     'arcParams': {'start': {'x': 0.16666649999141772,
       'y': -0.16666649999141778},
      'end': {'x': 0.1666664999914178, 'y': 0.16666649999141783}}}},
   {'circleArcEntity': {'center': {'x': 7.238184345098862e-18,
      'y': 3.626574587098518e-17},
     'circleParams': {'radius': 0.9999989999485066}}}]
}}


expected_sketch_obj = {
    'vertices': np.array([
        [0.16470588, 0.16470588],
        [-0.16470588, 0.16470588],
        [0.16470588, -0.16470588],
        [-0.16470588, -0.16470588],
        [-0.33333333, 0.],
        [0.33333333, 0.],
        [0., -1.],
        [1., 0.],
        [0., 1.],
        [-1., 0.],
    ]),
    'curves': [
        [0, 1],
        [2, 3],
        [1, 4, 3],
        [2, 5, 0],
        [6, 7, 8, 9],
    ],
}


class TestConvertDeepmindToObj(unittest.TestCase):
    def test_convert_sketch_to_obj(self):
        sketch_obj = convert_deepmind_to_obj.convert_sketch(dm_sketch)

        np.testing.assert_array_almost_equal(sketch_obj["vertices"], expected_sketch_obj["vertices"])
        self.assertEqual(sketch_obj["curves"], expected_sketch_obj["curves"])
