import unittest
import json
from pathlib import Path

from preprocess.deepmind_geometry import *
from preprocess.fusiongallery_geometry import *
from preprocess.convert_deepmind_to_fg import create_sketch_points, create_sketch_curves


class TestFusionGalleryDimension(unittest.TestCase):

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
    
    def load_data(self, dm_sketch_file, index):
        with open(dm_sketch_file) as f:
            dm_sketch_data = json.load(f)
        dm_entities = dm_sketch_data["entitySequence"]["entities"]
        dm_constraints = dm_sketch_data["constraintSequence"]["constraints"]
        points, point_map = create_sketch_points(dm_entities)
        curves, entity_map = create_sketch_curves(dm_entities, point_map)
        # Set the class variables with a dynamic index
        setattr(self, f"dm_entities{index}", dm_entities)
        setattr(self, f"dm_constraints{index}", dm_constraints)
        setattr(self, f"points{index}", points)
        setattr(self, f"point_map{index}", point_map)
        setattr(self, f"curves{index}", curves)
        setattr(self, f"entity_map{index}", entity_map)

    def test_distance_dimension(self):
        cst = self.dm_constraints0[16]
        fg_cst = FusionGalleryDimension(cst, self.points0, self.curves0, self.entity_map0)
        fg_cst_dict = fg_cst.to_dict()
        self.assertIsInstance(fg_cst_dict, dict)
        self.assertEqual(fg_cst_dict["type"], "SketchLinearDimension")
        self.assertIn("entity_one", fg_cst_dict)
        self.assertIn("entity_two", fg_cst_dict)
        entity_one_found = fg_cst_dict["entity_one"] in self.curves0 or fg_cst_dict["entity_one"] in self.points0
        entity_two_found = fg_cst_dict["entity_two"] in self.curves0 or fg_cst_dict["entity_two"] in self.points0
        self.assertTrue(entity_one_found)
        self.assertTrue(entity_two_found)
        self.assertEqual(fg_cst_dict["orientation"], "AlignedDimensionOrientation")
        self.assertAlmostEqual(fg_cst_dict["parameter"]["value"], 1.1956196943234358)

    def test_length_dimension(self):
        cst = self.dm_constraints0[18]
        fg_cst = FusionGalleryDimension(cst, self.points0, self.curves0, self.entity_map0)
        fg_cst_dict = fg_cst.to_dict()
        self.assertIsInstance(fg_cst_dict, dict)
        self.assertEqual(fg_cst_dict["type"], "SketchLinearDimension")
        self.assertIn("entity_one", fg_cst_dict)
        self.assertIn("entity_two", fg_cst_dict)
        self.assertIn(fg_cst_dict["entity_one"], self.points0)
        self.assertIn(fg_cst_dict["entity_two"], self.points0)
        self.assertAlmostEqual(fg_cst_dict["parameter"]["value"], 1.4945246179042941)

    def test_diameter_dimension_default(self):
        cst = self.dm_constraints2[5]
        fg_cst = FusionGalleryDimension(cst, self.points2, self.curves2, self.entity_map2)
        fg_cst_dict = fg_cst.to_dict()
        self.assertIsInstance(fg_cst_dict, dict)
        self.assertEqual(fg_cst_dict["type"], "SketchDiameterDimension")
        self.assertIn("curve", fg_cst_dict)
        self.assertIn(fg_cst_dict["curve"], self.curves2)
        self.assertAlmostEqual(fg_cst_dict["parameter"]["value"], 1.9999979992019683)

    def test_diameter_dimension(self):
        cst = self.dm_constraints6[26]
        fg_cst = FusionGalleryDimension(cst, self.points6, self.curves6, self.entity_map6)
        fg_cst_dict = fg_cst.to_dict()
        self.assertIsInstance(fg_cst_dict, dict)
        self.assertEqual(fg_cst_dict["type"], "SketchDiameterDimension")
        self.assertIn("curve", fg_cst_dict)
        self.assertIn(fg_cst_dict["curve"], self.curves6)
        self.assertAlmostEqual(fg_cst_dict["parameter"]["value"], 1.594244859324253)

    def test_radius_dimension(self):
        cst = self.dm_constraints6[32]
        fg_cst = FusionGalleryDimension(cst, self.points6, self.curves6, self.entity_map6)
        fg_cst_dict = fg_cst.to_dict()
        self.assertIsInstance(fg_cst_dict, dict)
        self.assertEqual(fg_cst_dict["type"], "SketchRadialDimension")
        self.assertIn("curve", fg_cst_dict)
        self.assertIn(fg_cst_dict["curve"], self.curves6)
        self.assertAlmostEqual(fg_cst_dict["parameter"]["value"], 0.1138746328088752)

    def test_angle_dimension(self):
        cst = self.dm_constraints8[6]
        fg_cst = FusionGalleryDimension(cst, self.points8, self.curves8, self.entity_map8)
        fg_cst_dict = fg_cst.to_dict()
        self.assertIsInstance(fg_cst_dict, dict)
        self.assertEqual(fg_cst_dict["type"], "SketchAngularDimension")
        self.assertIn("line_one", fg_cst_dict)
        self.assertIn("line_two", fg_cst_dict)
        self.assertIn(fg_cst_dict["line_one"], self.curves8)
        self.assertIn(fg_cst_dict["line_two"], self.curves8)
        self.assertAlmostEqual(fg_cst_dict["parameter"]["value"], 4.756022211684548)

    def test_angle_dimension_default_second(self):
        cst = self.dm_constraints8[7]
        fg_cst = FusionGalleryDimension(cst, self.points8, self.curves8, self.entity_map8)
        fg_cst_dict = fg_cst.to_dict()
        self.assertIsInstance(fg_cst_dict, dict)
        self.assertEqual(fg_cst_dict["type"], "SketchAngularDimension")
        self.assertIn("line_one", fg_cst_dict)
        self.assertIn("line_two", fg_cst_dict)
        self.assertIn(fg_cst_dict["line_one"], self.curves8)
        self.assertIn(fg_cst_dict["line_two"], self.curves8)
        self.assertAlmostEqual(fg_cst_dict["parameter"]["value"], 4.756022211684548)
