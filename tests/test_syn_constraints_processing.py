import json
import unittest

from preprocess.syn_contraints_preprocess import (
    process_for_syn_constraints,
    constraints_to_string, constraints_from_string,
    pp_constraints_to_string, pp_constraints_from_string,
    constraints_to_string_schema2, constraints_from_string_schema2,
)
import numpy as np


class TestSynConstraintsDataModule(unittest.TestCase):
    def test_syn_constraints_conversion(self):
        path = "syn_constraints_test_data/val.json"
        with open(path, "r") as f:
            data = json.load(f)
        data = data["data"]

        for example in data:
            constraints = example["constraints"]
            constraint_string = constraints_to_string(constraints)
            res = constraints_from_string(constraint_string)
            self.assertEqual(constraints, res)

    def test_syn_constraints_pp_conversion(self):
        path = "syn_constraints_test_data/val.json"
        with open(path, "r") as f:
            data = json.load(f)
        data = data["data"]

        for example in data:
            example = process_for_syn_constraints(example, return_mid_points=True)

            constraint_string = pp_constraints_to_string(constraints=example["constraints"],
                                                         mid_points=example["mid_points"])
            res = pp_constraints_from_string(constraint_string, example["mid_points"])

            print("constraints\n", example["constraints"])
            print("result\n", res)

            self.assertEqual(example["constraints"], res)

    def test_syn_constraints_schema2_conversion(self):
        # Sanity check
        # Bad input should return empty constraints, not raise exception
        pp_constraints_from_string("<0>")

        path = "syn_constraints_test_data/val.json"
        with open(path, "r") as f:
            data = json.load(f)
        data = data["data"]

        for example in data:
            constraints = example["constraints"]
            constraint_string = constraints_to_string_schema2(constraints)
            res = constraints_from_string_schema2(constraint_string)
            self.assertEqual(constraints, res)
