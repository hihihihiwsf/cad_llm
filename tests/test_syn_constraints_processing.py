import unittest
import json

from preprocess.syn_contraints_preprocess import constraints_to_string, constraints_from_string
from preprocess.syn_contraints_preprocess import get_pp_constraints_string, pp_constraints_from_string


class TestSynConstraintsDataModule(unittest.TestCase):
    def test_syn_constraints_conversion(self):
        path = "syn_constraints_test_data/val.json"
        with open(path, "r") as f:
            data = json.load(f)
        data = data["data"]

        for example in data:
            constraints = example["constraints"]
            contraint_string = constraints_to_string(constraints)
            res = constraints_from_string(contraint_string)
            self.assertEqual(constraints, res)

    def test_syn_constraints_pp_conversion(self):
        # Sanity check
        # Bad input should return empty constraints, not raise exception
        pp_constraints_from_string("<0>")

        path = "syn_constraints_test_data/val.json"
        with open(path, "r") as f:
            data = json.load(f)
        data = data["data"]

        for example in data:
            contraint_string = get_pp_constraints_string(example)
            res = pp_constraints_from_string(contraint_string)

            print("constraints\n", example["constraints"])
            print("result\n", res)
