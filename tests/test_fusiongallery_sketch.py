"""

Test a Fusion Gallery Sketch for self-consistency

"""


import math
from uuid import UUID
import unittest

from preprocess.fusiongallery_geometry.constraint import FusionGalleryConstraint
from preprocess.fusiongallery_geometry.dimension import FusionGalleryDimension

class TestFusionGallerySketch(unittest.TestCase):

    def run_sketch_test(self, sketch):
        self.assertIsInstance(sketch, dict)
        self.run_points_test(sketch)
        self.run_curves_test(sketch)
        self.run_constraints_test(sketch)
        self.run_dimensions_test(sketch)

    def run_points_test(self, sketch):
        self.assertIn("points", sketch)
        points = sketch["points"]
        for pt_uuid, pt in points.items():
            self.run_point_test(pt)
            self.assertTrue(self.is_valid_uuid(pt_uuid))

    def run_point_test(self, pt):
        self.assertEqual(pt["type"], "Point3D")
        self.assertFalse(math.isnan(pt["x"]))
        self.assertFalse(math.isnan(pt["y"]))
        self.assertEqual(pt["z"], 0)
    
    def run_curves_test(self, sketch):
        self.assertIn("curves", sketch)
        curves = sketch["curves"]
        for curve_uuid, curve in curves.items():
            self.run_curve_test(curve, sketch["points"])
            self.assertTrue(self.is_valid_uuid(curve_uuid))

    def run_curve_test(self, curve, points):
        types = {"SketchLine", "SketchArc", "SketchCircle"}
        self.assertIn(curve["type"], types)
        self.assertIn("construction_geom", curve)
        self.assertIsInstance(curve["construction_geom"], bool)
        if curve["type"] == "SketchLine" or curve["type"] == "SketchArc":
            self.assertIn(curve["start_point"], points)
            self.assertIn(curve["end_point"], points)
        if curve["type"] == "SketchCurve" or curve["type"] == "SketchArc":
            self.assertIn(curve["center_point"], points)
        
    def run_constraints_test(self, sketch):
        self.assertIn("constraints", sketch)
        constraints = sketch["constraints"]
        for constraint_uuid, constraint in constraints.items():
            self.run_constraint_test(constraint, sketch["curves"], sketch["points"])
            self.assertTrue(self.is_valid_uuid(constraint_uuid))

    def run_constraint_test(self, constraint, curves, points):
        self.assertIn(constraint["type"], FusionGalleryConstraint.types)
        uuid_keys = {
            "entity"
        }
        uuid_point_keys = {
            "point",
            "point_one",
            "point_two",
        }
        uuid_curve_keys = {
            "curve",
            "curve_one",
            "curve_two",
            "line_one",
            "line_two",
            "line",
            "mid_point_curve"
        }
        # Check that if there are references to uuids, then they actually 
        # reference a curve or point
        for key, val in constraint.items():
            if key in uuid_keys:
                in_curves = val in curves
                in_points = val in points
                self.assertTrue(in_curves or in_points)
            if key in uuid_point_keys:
                self.assertIn(val, points)
            if key in uuid_curve_keys:
                self.assertIn(val, curves)

    def run_dimensions_test(self, sketch):
        self.assertIn("dimensions", sketch)
        dimensions = sketch["dimensions"]
        for dimension_uuid, dimension in dimensions.items():
            self.run_dimension_test(dimension, sketch["curves"], sketch["points"])
            self.assertTrue(self.is_valid_uuid(dimension_uuid))

    def run_dimension_test(self, dimension, curves, points):
        self.assertIn(dimension["type"], FusionGalleryDimension.types)
        uuid_keys = {
            "entity_one",
            "entity_two"
        }
        uuid_point_keys = {
            "point",
            "point_one",
            "point_two",
        }
        uuid_curve_keys = {
            "curve",
            "curve_one",
            "curve_two",
            "line_one",
            "line_two",
            "line",
        }
        # Check that if there are references to uuids, then they actually 
        # reference a curve or point
        for key, val in dimension.items():
            if key in uuid_keys:
                in_curves = val in curves
                in_points = val in points
                self.assertTrue(in_curves or in_points)
            if key in uuid_point_keys:
                self.assertIn(val, points)
            if key in uuid_curve_keys:
                self.assertIn(val, curves)


    def is_valid_uuid(self, uuid_to_test, version=1):
        """
        Check if uuid_to_test is a valid UUID.
        
        Parameters
        ----------
        uuid_to_test : str
        version : {1, 2, 3, 4}
        
        Returns
        -------
        `True` if uuid_to_test is a valid UUID, otherwise `False`.
        
        Examples
        --------
        >>> is_valid_uuid('c9bf9e57-1685-4c89-bafb-ff5af830be8a')
        True
        >>> is_valid_uuid('c9bf9e58')
        False
        """
        
        try:
            uuid_obj = UUID(uuid_to_test, version=version)
        except ValueError:
            return False
        return str(uuid_obj) == uuid_to_test