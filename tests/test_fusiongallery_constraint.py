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
        print("Sss")

    def test_coincident_to_collinear_constraint(self):
        cst = self.dm_constraints[0]
        fg_cst = FusionGalleryConstraint(cst, self.points, self.curves, self.entity_map)
        fg_cst_dict = fg_cst.to_dict()
        self.assertIsInstance(fg_cst_dict, dict)
        # Two lines become a collinear constraint
        self.assertEqual(fg_cst_dict["type"], "CollinearConstraint")
        self.assertIn("line_one", fg_cst_dict)
        self.assertIn("line_two", fg_cst_dict)
        self.assertIn(fg_cst_dict["line_one"], self.curves)
        self.assertIn(fg_cst_dict["line_two"], self.curves)
        # Check dm entities are also lines
        ent_0_index = cst["coincidentConstraint"]["entities"][0]
        ent_1_index = cst["coincidentConstraint"]["entities"][1]
        ent_0 = self.dm_entities[ent_0_index]
        ent_1 = self.dm_entities[ent_1_index]
        self.assertIn("lineEntity", ent_0)
        self.assertIn("lineEntity", ent_1)

    def test_coincident_constraint(self):
        cst = self.dm_constraints[1]
        fg_cst = FusionGalleryConstraint(cst, self.points, self.curves, self.entity_map)
        fg_cst_dict = fg_cst.to_dict()
        self.assertIsInstance(fg_cst_dict, dict)
     
    def test_coincident_three_entity_constraint(self):
        cst = self.dm_constraints[2]
        # This is malformed data sample, where the third index (23)
        # is out of range
        ent_2_index = cst["coincidentConstraint"]["entities"][0]
        self.assertNotIn(ent_2_index, self.dm_entities)
        # Check our class doesnt throw up with this bad data
        fg_cst = FusionGalleryConstraint(cst, self.points, self.curves, self.entity_map)
        fg_cst_dict = fg_cst.to_dict()
        self.assertIsInstance(fg_cst_dict, dict)
        self.assertEqual(fg_cst_dict["type"], "CoincidentConstraint")
        self.assertIn("entity", fg_cst_dict)
        self.assertIn("point", fg_cst_dict)
        self.assertIn(fg_cst_dict["entity"], self.curves)
        self.assertIn(fg_cst_dict["point"], self.points)
        # Check dm entities are the type we expect
        ent_0_index = cst["coincidentConstraint"]["entities"][0]
        ent_1_index = cst["coincidentConstraint"]["entities"][1]
        ent_0 = self.dm_entities[ent_0_index]
        ent_1 = self.dm_entities[ent_1_index]
        self.assertIn("pointEntity", ent_0)
        self.assertIn("lineEntity", ent_1)