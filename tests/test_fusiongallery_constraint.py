import unittest
import json
from pathlib import Path

from preprocess.deepmind_geometry import *
from preprocess.fusiongallery_geometry import *
from preprocess.convert_deepmind_to_fg import DeepmindToFusionGalleryConverter


class TestFusionGalleryConstraint(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load our test data and assign different numbered variables
        cls.load_data(cls, Path("tests/test_data/dm_sketch_0.json"), 0)
        cls.load_data(cls, Path("tests/test_data/dm_sketch_1.json"), 1)
        cls.load_data(cls, Path("tests/test_data/dm_sketch_2.json"), 2)
        cls.load_data(cls, Path("tests/test_data/dm_sketch_3.json"), 3)
        cls.load_data(cls, Path("tests/test_data/dm_sketch_4.json"), 4)
        cls.load_data(cls, Path("tests/test_data/dm_sketch_5.json"), 5)
        cls.load_data(cls, Path("tests/test_data/dm_sketch_6.json"), 6)
        cls.load_data(cls, Path("tests/test_data/dm_sketch_7.json"), 7)
        cls.load_data(cls, Path("tests/test_data/dm_sketch_8.json"), 8)
        cls.load_data(cls, Path("tests/test_data/dm_sketch_9.json"), 9)
        cls.load_data(cls, Path("tests/test_data/dm_sketch_10.json"), 10)
        cls.load_data(cls, Path("tests/test_data/dm_sketch_11.json"), 11)
    
    def load_data(self, dm_sketch_file, index):
        with open(dm_sketch_file) as f:
            dm_sketch_data = json.load(f)
        dm_entities = dm_sketch_data["entitySequence"]["entities"]
        dm_constraints = dm_sketch_data["constraintSequence"]["constraints"]
        points, point_map = DeepmindToFusionGalleryConverter.create_sketch_points(dm_entities)
        curves, entity_map = DeepmindToFusionGalleryConverter.create_sketch_curves(dm_entities, point_map)
        # Set the class variables with a dynamic index
        setattr(self, f"dm_entities{index}", dm_entities)
        setattr(self, f"dm_constraints{index}", dm_constraints)
        setattr(self, f"points{index}", points)
        setattr(self, f"point_map{index}", point_map)
        setattr(self, f"curves{index}", curves)
        setattr(self, f"entity_map{index}", entity_map)

    def test_coincident_constraint_merged_points(self):
        # This constraint is between identical points that meet to 
        # join together two lines
        cst = self.dm_constraints0[1]
        fg_cst = FusionGalleryConstraint(cst, self.points0, self.curves0, self.entity_map0)
        fg_cst_dict = fg_cst.to_dict()
        # We remove these types of constraints between merged points
        self.assertTrue(fg_cst_dict is None or fg_cst_dict == "Merge")

    def test_coincident_constraint_different_merged_points(self):
        # This constraint is between near identical points that we need to merge
        cst = self.dm_constraints0[0]
        fg_cst = FusionGalleryConstraint(cst, self.points0, self.curves0, self.entity_map0)
        fg_cst_dict = fg_cst.to_dict()
        # We remove these types of constraints between merged points
        self.assertTrue(fg_cst_dict is None or fg_cst_dict == "Merge")

    def test_coincident_constraint_three_merged_points(self):
        # This constraint is between near three identical points that we need to merge
        cst = self.dm_constraints0[2]
        fg_cst = FusionGalleryConstraint(cst, self.points0, self.curves0, self.entity_map0)
        fg_cst_dict = fg_cst.to_dict()
        # We remove these types of constraints between merged points
        self.assertTrue(fg_cst_dict is None or fg_cst_dict == "Merge")
    
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

    def test_horizontal_constraint_points(self):
        cst = self.dm_constraints3[5]
        fg_cst = FusionGalleryConstraint(cst, self.points3, self.curves3, self.entity_map3)
        fg_cst_dict = fg_cst.to_dict()
        self.assertIsNotNone(fg_cst_dict)
        self.assertIsInstance(fg_cst_dict, dict)
        self.assertEqual(fg_cst_dict["type"], "HorizontalPointsConstraint")
        self.assertIn("point_one", fg_cst_dict)
        self.assertIn("point_two", fg_cst_dict)
        self.assertIn(fg_cst_dict["point_one"], self.points3)
        self.assertIn(fg_cst_dict["point_two"], self.points3)

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
        
    def test_vertical_constraint_points(self):
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

    def test_vertical_constraint_multiple_lines(self):
        cst = self.dm_constraints9[8]
        fg_cst = FusionGalleryConstraint(cst, self.points9, self.curves9, self.entity_map9)
        fg_cst_list = fg_cst.to_dict()
        self.assertIsNotNone(fg_cst_list)
        # This constraint is applied to multiple lines
        # so we need to return multiple constraints
        self.assertIsInstance(fg_cst_list, list)
        for fg_cst in fg_cst_list:
            self.assertEqual(fg_cst["type"], "VerticalConstraint")
            self.assertIn("line", fg_cst)
            self.assertIn(fg_cst["line"], self.curves9)

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
        # Here we actually create a construction line and add it to the curves
        cst = self.dm_constraints0[13]
        prev_curve_count = len(self.curves0)
        fg_cst = FusionGalleryConstraint(cst, self.points0, self.curves0, self.entity_map0)
        fg_cst_dict = fg_cst.to_dict()
        new_curve_count = len(self.curves0)
        self.assertEqual(prev_curve_count, new_curve_count - 1)
        self.assertIsNotNone(fg_cst_dict)
        self.assertIsInstance(fg_cst_dict, dict)
        self.assertEqual(fg_cst_dict["type"], "MidPointConstraint")
        self.assertIn("point", fg_cst_dict)
        self.assertIn("mid_point_curve", fg_cst_dict)
        self.assertIn(fg_cst_dict["point"], self.points0)
        self.assertIn(fg_cst_dict["mid_point_curve"], self.curves0)

    def test_equal_constraint(self):
        cst = self.dm_constraints4[8]
        fg_cst = FusionGalleryConstraint(cst, self.points4, self.curves4, self.entity_map4)
        fg_cst_dict = fg_cst.to_dict()
        self.assertIsNotNone(fg_cst_dict)
        self.assertIsInstance(fg_cst_dict, dict)
        self.assertEqual(fg_cst_dict["type"], "EqualConstraint")
        self.assertIn("curve_one", fg_cst_dict)
        self.assertIn("curve_two", fg_cst_dict)
        self.assertIn(fg_cst_dict["curve_one"], self.curves4)
        self.assertIn(fg_cst_dict["curve_two"], self.curves4)

    def test_equal_constraint_multiple(self):
        # Handle multiple entity equal constraints
        # e.g.
        # 
        # "equalConstraint": {
        #     "entities": [
        #         0,
        #         2,
        #         4,
        #         6,
        #         14,
        #         12,
        #         10,
        #         8
        #     ]
        # }
        cst = self.dm_constraints5[5]
        fg_cst = FusionGalleryConstraint(cst, self.points5, self.curves5, self.entity_map5)
        fg_cst_list = fg_cst.to_dict()
        self.assertIsNotNone(fg_cst_list)
        # This constraint is applied to multiple line pairs
        # so we need to return multiple constraints
        self.assertIsInstance(fg_cst_list, list)
        # There should be a constraint between each entities
        self.assertEqual(len(fg_cst_list), len(cst['equalConstraint']['entities']) - 1)
        for fg_cst in fg_cst_list:
            self.assertEqual(fg_cst["type"], "EqualConstraint")
            self.assertIn("curve_one", fg_cst)
            self.assertIn("curve_two", fg_cst)
            self.assertIn(fg_cst["curve_one"], self.curves5)
            self.assertIn(fg_cst["curve_two"], self.curves5)

    def test_concentric_constraint(self):
        # Test concentric with two curves
        cst = self.dm_constraints6[23]
        fg_cst = FusionGalleryConstraint(cst, self.points6, self.curves6, self.entity_map6)
        fg_cst_dict = fg_cst.to_dict()
        self.assertIsNotNone(fg_cst_dict)
        self.assertIsInstance(fg_cst_dict, dict)
        self.assertEqual(fg_cst_dict["type"], "ConcentricConstraint")
        self.assertIn("curve_one", fg_cst_dict)
        self.assertIn("curve_two", fg_cst_dict)
        self.assertIn(fg_cst_dict["curve_one"], self.curves6)
        self.assertIn(fg_cst_dict["curve_two"], self.curves6)

    def test_concentric_constraint_points(self):
        # Test concentric with two points
        cst = self.dm_constraints7[9]
        fg_cst = FusionGalleryConstraint(cst, self.points7, self.curves7, self.entity_map7)
        fg_cst_dict = fg_cst.to_dict()
        self.assertIsNotNone(fg_cst_dict)
        self.assertIsInstance(fg_cst_dict, dict)
        self.assertEqual(fg_cst_dict["type"], "ConcentricConstraint")
        self.assertIn("curve_one", fg_cst_dict)
        self.assertIn("curve_two", fg_cst_dict)
        self.assertIn(fg_cst_dict["curve_one"], self.curves7)
        self.assertIn(fg_cst_dict["curve_two"], self.curves7)
    
    def test_mirror_constraint(self):
        cst = self.dm_constraints10[6]
        fg_cst = FusionGalleryConstraint(cst, self.points10, self.curves10, self.entity_map10)
        fg_cst_list = fg_cst.to_dict()
        self.assertIsNotNone(fg_cst_list)
        self.assertIsInstance(fg_cst_list, list)
        self.assertEqual(len(fg_cst_list), 2)
        for fg_cst in fg_cst_list:
            self.assertEqual(fg_cst["type"], "SymmetryConstraint")
            self.assertIn("entity_one", fg_cst)
            self.assertIn("entity_two", fg_cst)
            entity_one_found = fg_cst["entity_one"] in self.curves10 or fg_cst["entity_one"] in self.points10
            entity_two_found = fg_cst["entity_two"] in self.curves10 or fg_cst["entity_two"] in self.points10
            self.assertTrue(entity_one_found)
            self.assertTrue(entity_two_found)

    def test_fix_constraint(self):
        cst = self.dm_constraints11[0]
        fg_cst = FusionGalleryConstraint(cst, self.points11, self.curves11, self.entity_map11)
        fg_cst_dict = fg_cst.to_dict()
        self.assertEqual(fg_cst_dict, "Fix")
        entity_indices = cst["fixConstraint"]["entities"]
        for index in entity_indices:
            entity = self.entity_map11[index]
            # Some entities are points
            if entity["type"] == "curve":
                self.assertIn(entity["uuid"], self.curves11)
                curve = self.curves11[entity["uuid"]]
                self.assertTrue(curve["fixed"])
            elif entity["type"] == "point":
                self.assertIn(entity["uuid"], self.points11)
                point = self.points11[entity["uuid"]]
                self.assertTrue(point["fixed"])




