import unittest
import json
from pathlib import Path

from preprocess.deepmind_geometry import *
from preprocess.fusiongallery_geometry import *
from preprocess.convert_deepmind_to_fg import create_sketch_points, create_sketch_curves


class TestFusionGalleryConstraint(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dm_sketch_file = Path("tests/test_data/dm_sketch_0.json")
        with open(dm_sketch_file) as f:
            cls.dm_sketch_data = json.load(f)
        cls.dm_entities = cls.dm_sketch_data["entitySequence"]["entities"]
        cls.dm_constraints = cls.dm_sketch_data["constraintSequence"]["constraints"]
        cls.points, cls.point_map = create_sketch_points(cls.dm_entities)
        cls.curves, cls.entity_map = create_sketch_curves(cls.dm_entities, cls.point_map)

    def test_coincident_constraint_merged_points(self):
        # This constraint is between identical points that meet to 
        # join together two lines
        cst = self.dm_constraints[1]
        fg_cst = FusionGalleryConstraint(cst, self.points, self.curves, self.entity_map)
        fg_cst_dict = fg_cst.to_dict()
        # We remove these types of constraints between merged points
        self.assertIsNone(fg_cst_dict)

    def test_coincident_constraint_different_merged_points(self):
        # This constraint is between near identical points that we need to merge
        cst = self.dm_constraints[0]
        fg_cst = FusionGalleryConstraint(cst, self.points, self.curves, self.entity_map)
        fg_cst_dict = fg_cst.to_dict()
        # We remove these types of constraints between merged points
        self.assertIsNone(fg_cst_dict)

    def test_coincident_constraint_three_merged_points(self):
        # This constraint is between near three identical points that we need to merge
        cst = self.dm_constraints[2]
        fg_cst = FusionGalleryConstraint(cst, self.points, self.curves, self.entity_map)
        fg_cst_dict = fg_cst.to_dict()
        # We remove these types of constraints between merged points
        self.assertIsNone(fg_cst_dict)
    
    def test_parallel_constraint(self):
        cst = self.dm_constraints[8]
        fg_cst = FusionGalleryConstraint(cst, self.points, self.curves, self.entity_map)
        fg_cst_dict = fg_cst.to_dict()
        self.assertIsNotNone(fg_cst_dict)
        self.assertIsInstance(fg_cst_dict, dict)
        self.assertEqual(fg_cst_dict["type"], "ParallelConstraint")
        self.assertIn("line_one", fg_cst_dict)
        self.assertIn("line_two", fg_cst_dict)
        self.assertIn(fg_cst_dict["line_one"], self.curves)
        self.assertIn(fg_cst_dict["line_two"], self.curves)
        
