import unittest
import json

from preprocess.syn_contraints_preprocess import constraints_to_string, constraints_from_string


class TestSynConstraintsDataModule(unittest.TestCase):
    def test_syn_constraints_datamodule(self):
        path = "syn_constraints_test_data/val.json"
        with open(path, "r") as f:
            data = json.load(f)
        data = data["data"]

        for example in data:
            constraints = example["constraints"]
            contraint_string = constraints_to_string(constraints)
            res = constraints_from_string(contraint_string)
            self.assertEqual(constraints, res)
