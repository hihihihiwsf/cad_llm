import unittest
import json
from pathlib import Path

from preprocess.deepmind_geometry import *
from preprocess.fusiongallery_geometry import *
from preprocess.convert_deepmind_to_fg import create_sketch_points, create_sketch_curves


class TestFusionGalleryConstraint(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dm_sketch_file0 = Path("tests/test_data/dm_sketch_0.json")
        with open(dm_sketch_file0) as f:
            cls.dm_sketch_data0 = json.load(f)
        cls.dm_entities0 = cls.dm_sketch_data0["entitySequence"]["entities"]
        cls.dm_constraints0 = cls.dm_sketch_data0["constraintSequence"]["constraints"]
        cls.points0, cls.point_map0 = create_sketch_points(cls.dm_entities0)
        cls.curves0, cls.entity_map0 = create_sketch_curves(cls.dm_entities0, cls.point_map0)

        dm_sketch_file1 = Path("tests/test_data/dm_sketch_1.json")
        with open(dm_sketch_file1) as f:
            cls.dm_sketch_data1 = json.load(f)
        cls.dm_entities1 = cls.dm_sketch_data1["entitySequence"]["entities"]
        cls.dm_constraints1 = cls.dm_sketch_data1["constraintSequence"]["constraints"]
        cls.points1, cls.point_map1 = create_sketch_points(cls.dm_entities1)
        cls.curves1, cls.entity_map1 = create_sketch_curves(cls.dm_entities1, cls.point_map1)

        dm_sketch_file2 = Path("tests/test_data/dm_sketch_2.json")
        with open(dm_sketch_file2) as f:
            cls.dm_sketch_data2 = json.load(f)
        cls.dm_entities2 = cls.dm_sketch_data2["entitySequence"]["entities"]
        cls.dm_constraints2 = cls.dm_sketch_data2["constraintSequence"]["constraints"]
        cls.points2, cls.point_map2 = create_sketch_points(cls.dm_entities2)
        cls.curves2, cls.entity_map2 = create_sketch_curves(cls.dm_entities2, cls.point_map2)

    def test_coincident_constraint_merged_points(self):
        # This constraint is between identical points that meet to 
        # join together two lines
        cst = self.dm_constraints0[1]
        fg_cst = FusionGalleryConstraint(cst, self.points0, self.curves0, self.entity_map0)
        fg_cst_dict = fg_cst.to_dict()
        # We remove these types of constraints between merged points
        self.assertIsNone(fg_cst_dict)

    def test_coincident_constraint_different_merged_points(self):
        # This constraint is between near identical points that we need to merge
        cst = self.dm_constraints0[0]
        fg_cst = FusionGalleryConstraint(cst, self.points0, self.curves0, self.entity_map0)
        fg_cst_dict = fg_cst.to_dict()
        # We remove these types of constraints between merged points
        self.assertIsNone(fg_cst_dict)

    def test_coincident_constraint_three_merged_points(self):
        # This constraint is between near three identical points that we need to merge
        cst = self.dm_constraints0[2]
        fg_cst = FusionGalleryConstraint(cst, self.points0, self.curves0, self.entity_map0)
        fg_cst_dict = fg_cst.to_dict()
        # We remove these types of constraints between merged points
        self.assertIsNone(fg_cst_dict)
    
    def test_parallel_constraint(self):
        cst = self.dm_constraints0[8]
        fg_cst = FusionGalleryConstraint(cst, self.points0, self.curves0, self.entity_map0)
        fg_cst_dict = fg_cst.to_dict()
        self.assertIsNotNone(fg_cst_dict)
        self.assertIsInstance(fg_cst_dict, dict)
        self.assertEqual(fg_cst_dict["type"], "ParallelConstraint")
        self.assertIn("line_one", fg_cst_dict)
        self.assertIn("line_two", fg_cst_dict)
        self.assertIn(fg_cst_dict["line_one"], self.curves0)
        self.assertIn(fg_cst_dict["line_two"], self.curves0)

    def test_horizontal_constraint_line(self):
        cst = self.dm_constraints0[11]
        fg_cst = FusionGalleryConstraint(cst, self.points0, self.curves0, self.entity_map0)
        fg_cst_dict = fg_cst.to_dict()
        self.assertIsNotNone(fg_cst_dict)
        self.assertIsInstance(fg_cst_dict, dict)
        self.assertEqual(fg_cst_dict["type"], "HorizontalConstraint")
        self.assertIn("line", fg_cst_dict)
        self.assertIn(fg_cst_dict["line"], self.curves0)

    def test_horizontal_constraint_multiple_lines(self):
        cst = self.dm_constraints1[11]
        fg_cst = FusionGalleryConstraint(cst, self.points1, self.curves1, self.entity_map1)
        fg_cst_list = fg_cst.to_dict()
        self.assertIsNotNone(fg_cst_list)
        # This constraint is applied to multiple lines
        # so we need to return multiple constraints
        self.assertIsInstance(fg_cst_list, list)
        for fg_cst in fg_cst_list:
            self.assertEqual(fg_cst["type"], "HorizontalConstraint")
            self.assertIn("line", fg_cst)
            self.assertIn(fg_cst["line"], self.curves1)
        
    def test_vertical_points_constraint(self):
        cst = self.dm_constraints0[12]
        fg_cst = FusionGalleryConstraint(cst, self.points0, self.curves0, self.entity_map0)
        fg_cst_dict = fg_cst.to_dict()
        self.assertIsNotNone(fg_cst_dict)
        self.assertIsInstance(fg_cst_dict, dict)
        self.assertEqual(fg_cst_dict["type"], "VerticalPointsConstraint")
        self.assertIn("point_one", fg_cst_dict)
        self.assertIn("point_two", fg_cst_dict)
        self.assertIn(fg_cst_dict["point_one"], self.points0)
        self.assertIn(fg_cst_dict["point_two"], self.points0)

    def test_vertical_constraint_line(self):
        cst = self.dm_constraints1[12]
        fg_cst = FusionGalleryConstraint(cst, self.points1, self.curves1, self.entity_map1)
        fg_cst_dict = fg_cst.to_dict()
        self.assertIsNotNone(fg_cst_dict)
        self.assertIsInstance(fg_cst_dict, dict)
        self.assertEqual(fg_cst_dict["type"], "VerticalConstraint")
        self.assertIn("line", fg_cst_dict)
        self.assertIn(fg_cst_dict["line"], self.curves1)

    def test_tangent_constraint(self):
        cst = self.dm_constraints2[6]
        fg_cst = FusionGalleryConstraint(cst, self.points2, self.curves2, self.entity_map2)
        fg_cst_dict = fg_cst.to_dict()
        self.assertIsNotNone(fg_cst_dict)
        self.assertIsInstance(fg_cst_dict, dict)
        self.assertEqual(fg_cst_dict["type"], "TangentConstraint")
        self.assertIn("curve_one", fg_cst_dict)
        self.assertIn("curve_two", fg_cst_dict)
        self.assertIn(fg_cst_dict["curve_one"], self.curves2)
        self.assertIn(fg_cst_dict["curve_two"], self.curves2)

    def test_perpendicular_constraint(self):
        cst = self.dm_constraints0[15]
        fg_cst = FusionGalleryConstraint(cst, self.points0, self.curves0, self.entity_map0)
        fg_cst_dict = fg_cst.to_dict()
        self.assertIsNotNone(fg_cst_dict)
        self.assertIsInstance(fg_cst_dict, dict)
        self.assertEqual(fg_cst_dict["type"], "PerpendicularConstraint")
        self.assertIn("line_one", fg_cst_dict)
        self.assertIn("line_two", fg_cst_dict)
        self.assertIn(fg_cst_dict["line_one"], self.curves0)
        self.assertIn(fg_cst_dict["line_two"], self.curves0)        

    def test_midpoint_constraint(self):
        cst = self.dm_constraints2[7]
        fg_cst = FusionGalleryConstraint(cst, self.points2, self.curves2, self.entity_map2)
        fg_cst_dict = fg_cst.to_dict()
        self.assertIsNotNone(fg_cst_dict)
        self.assertIsInstance(fg_cst_dict, dict)
        self.assertEqual(fg_cst_dict["type"], "MidPointConstraint")
        self.assertIn("point", fg_cst_dict)
        self.assertIn("mid_point_curve", fg_cst_dict)
        self.assertIn(fg_cst_dict["point"], self.points2)
        self.assertIn(fg_cst_dict["mid_point_curve"], self.curves2)

    def test_midpoint_constraint_endpoints(self):
        cst = self.dm_constraints0[13]
        fg_cst = FusionGalleryConstraint(cst, self.points0, self.curves0, self.entity_map0)
        fg_cst_dict = fg_cst.to_dict()
        self.assertIsNotNone(fg_cst_dict)
        self.assertIsInstance(fg_cst_dict, dict)
        self.assertEqual(fg_cst_dict["type"], "MidPointConstraint")
        # TODO: Handle this case that has endpoints
