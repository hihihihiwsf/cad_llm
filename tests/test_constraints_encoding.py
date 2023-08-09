import unittest
import json
from preprocess.preprocess_syn_constraints import constraints_to_string, constraints_from_string


class TestSynConstraintsEncoding(unittest.TestCase):
    def test_syn_constraints_encoding(self):
        path = "syn_constraints_test_data/val.json"
        with open(path, "r") as f:
            data = json.load(f)

        for example in data["data"]:
            constraints = example["constraints"]
            constraints_str = constraints_to_string(constraints)
            res = constraints_from_string(constraints_str)
            self.assertEqual(constraints, res)
