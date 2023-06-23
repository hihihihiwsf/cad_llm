import unittest
from preprocess.deepmind_geometry import *
from preprocess.fusiongallery_geometry import *


class TestFusionGalleryGeometry(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.line = {
            "lineEntity": {
                "start": {
                    "x": 0.0909089999989889,
                    "y": 0.1515149999983142
                },
                "end": {
                    "x": 0.030302999999663023,
                    "y": 0.1515149999983142
                }
            }
        }
        cls.arc = {
            "circleArcEntity": {
                "center": {
                    "x": 0.9444434999904506,
                    "y": 0.4999994999949444
                },
                "arcParams": {
                    "start": {
                        "x": 0.9999989999898888,
                        "y": 0.4999994999949444
                    },
                    "end": {
                        "x": 0.9444434999904505,
                        "y": 0.5555549999943826
                    }
                }
            }
        }
        cls.arc_clockwise = {
            "circleArcEntity": {
                "center": {
                    "x": 0.9999989999743334,
                    "y": 0.6133327199842576
                },
                "arcParams": {
                    "start": {
                        "x": 0.9999989999743334,
                        "y": 0.3770659458864415
                    },
                    "end": {
                        "x": 0.7637322258765172,
                        "y": 0.6133327199842576
                    },
                    "isClockwise": True
                }
            }
        }  
        cls.circle = {
            "circleArcEntity": {
                "center": {
                    "x": -0.9529827721390653,
                    "y": -1.570780979108142e-16
                },
                "circleParams": {
                    "radius": 0.04701622784746812
                }
            }
        }
        cls.point_map = FusionGalleryPointMap([
            cls.line, cls.circle, cls.arc, cls.arc_clockwise
        ])
    
    def test_point_map(self):
        pm = FusionGalleryPointMap([
            self.line, self.circle, self.arc
        ])
        self.assertIsNotNone(pm)
        self.assertIsInstance(pm.map, dict)
        self.assertEqual(len(pm.map), 6)
        # Check we are removing duplicates from the line
        pm_dupe = FusionGalleryPointMap([
            self.line, self.line, self.circle, self.arc
        ])
        self.assertEqual(len(pm_dupe.map), 6)

    def test_point(self):
        start = self.line["lineEntity"]["start"]
        dm_start = DeepmindPoint(start)
        fg_start = FusionGalleryPoint(dm_start)
        self.assertIsNotNone(fg_start)
        self.assertAlmostEqual(start["x"], fg_start.x)
        self.assertAlmostEqual(start["x"], fg_start.point[0])
        self.assertAlmostEqual(start["y"], fg_start.y)
        self.assertAlmostEqual(start["y"], fg_start.point[1])
        # Dict structure
        fg_start_dict = fg_start.to_dict()
        self.assertEqual(fg_start_dict["type"], "Point3D")
        self.assertAlmostEqual(fg_start_dict["x"], fg_start.x)
        self.assertAlmostEqual(fg_start_dict["y"], fg_start.y)
        self.assertAlmostEqual(fg_start_dict["z"], 0)

    def test_line(self):
        start = self.line["lineEntity"]["start"]
        end = self.line["lineEntity"]["end"]
        dm_line = DeepmindLine(self.line["lineEntity"])
        fg_line = FusionGalleryLine(dm_line, self.point_map.map)
        self.assertIsNotNone(fg_line)
        self.assertAlmostEqual(start["x"], fg_line.start.x)
        self.assertAlmostEqual(start["y"], fg_line.start.y)
        self.assertAlmostEqual(end["x"], fg_line.end.x)
        self.assertAlmostEqual(end["y"], fg_line.end.y)
        # Dict structure
        fg_line_dict = fg_line.to_dict()
        self.assertEqual(fg_line_dict["type"], "SketchLine")
        self.assertEqual(fg_line_dict["construction_geom"], fg_line.is_construction)
        start_uuid = fg_line_dict["start_point"]
        end_uuid = fg_line_dict["end_point"]
        start_uuid_point = self.point_map.uuid_map[start_uuid]
        end_uuid_point = self.point_map.uuid_map[end_uuid]
        self.assertAlmostEqual(start["x"], start_uuid_point.x)
        self.assertAlmostEqual(start["y"], start_uuid_point.y)
        self.assertAlmostEqual(end["x"], end_uuid_point.x)
        self.assertAlmostEqual(end["y"], end_uuid_point.y)

    def test_arc(self):
        center = self.arc["circleArcEntity"]["center"]
        start = self.arc["circleArcEntity"]["arcParams"]["start"]
        end = self.arc["circleArcEntity"]["arcParams"]["end"]
        dm_arc = DeepmindArc(self.arc["circleArcEntity"])
        fg_arc = FusionGalleryArc(dm_arc, self.point_map.map)
        self.assertIsNotNone(fg_arc)
        self.assertAlmostEqual(center["x"], fg_arc.center.x)
        self.assertAlmostEqual(center["y"], fg_arc.center.y)
        self.assertAlmostEqual(start["x"], fg_arc.start.x)
        self.assertAlmostEqual(start["y"], fg_arc.start.y)
        self.assertAlmostEqual(end["x"], fg_arc.end.x)
        self.assertAlmostEqual(end["y"], fg_arc.end.y)
        # Dict structure
        fg_arc_dict = fg_arc.to_dict()
        self.assertEqual(fg_arc_dict["type"], "SketchArc")
        self.assertEqual(fg_arc_dict["construction_geom"], fg_arc.is_construction)
        center_uuid = fg_arc_dict["center_point"]
        start_uuid = fg_arc_dict["start_point"]
        end_uuid = fg_arc_dict["end_point"]
        center_uuid_point = self.point_map.uuid_map[center_uuid]
        start_uuid_point = self.point_map.uuid_map[start_uuid]
        end_uuid_point = self.point_map.uuid_map[end_uuid]
        self.assertAlmostEqual(center["x"], center_uuid_point.x)
        self.assertAlmostEqual(center["y"], center_uuid_point.y)
        self.assertAlmostEqual(start["x"], start_uuid_point.x)
        self.assertAlmostEqual(start["y"], start_uuid_point.y)
        self.assertAlmostEqual(end["x"], end_uuid_point.x)
        self.assertAlmostEqual(end["y"], end_uuid_point.y)
        # TODO: other tests on the arc geometry e.g. ref vector etc


    def test_arc_clockwise(self):
        center = self.arc_clockwise["circleArcEntity"]["center"]
        start = self.arc_clockwise["circleArcEntity"]["arcParams"]["start"]
        end = self.arc_clockwise["circleArcEntity"]["arcParams"]["end"]
        dm_arc = DeepmindArc(self.arc_clockwise["circleArcEntity"])
        fg_arc = FusionGalleryArc(dm_arc, self.point_map.map)
        self.assertIsNotNone(fg_arc)
        self.assertAlmostEqual(center["x"], fg_arc.center.x)
        self.assertAlmostEqual(center["y"], fg_arc.center.y)
        self.assertAlmostEqual(start["x"], fg_arc.start.x)
        self.assertAlmostEqual(start["y"], fg_arc.start.y)
        self.assertAlmostEqual(end["x"], fg_arc.end.x)
        self.assertAlmostEqual(end["y"], fg_arc.end.y)
        # Dict structure
        # Start and end points are flipped in the dict
        fg_arc_dict = fg_arc.to_dict()
        self.assertEqual(fg_arc_dict["type"], "SketchArc")
        self.assertEqual(fg_arc_dict["construction_geom"], fg_arc.is_construction)
        center_uuid = fg_arc_dict["center_point"]
        start_uuid = fg_arc_dict["start_point"]
        end_uuid = fg_arc_dict["end_point"]
        center_uuid_point = self.point_map.uuid_map[center_uuid]
        start_uuid_point = self.point_map.uuid_map[start_uuid]
        end_uuid_point = self.point_map.uuid_map[end_uuid]
        self.assertAlmostEqual(center["x"], center_uuid_point.x)
        self.assertAlmostEqual(center["y"], center_uuid_point.y)
        self.assertAlmostEqual(start["x"], end_uuid_point.x)
        self.assertAlmostEqual(start["y"], end_uuid_point.y)
        self.assertAlmostEqual(end["x"], start_uuid_point.x)
        self.assertAlmostEqual(end["y"], start_uuid_point.y)


    def test_circle(self):
        center = self.circle["circleArcEntity"]["center"]
        radius = self.circle["circleArcEntity"]["circleParams"]["radius"]
        dm_circle = DeepmindCircle(self.circle["circleArcEntity"])
        fg_circle = FusionGalleryCircle(dm_circle, self.point_map.map)
        self.assertIsNotNone(fg_circle)
        self.assertAlmostEqual(center["x"], fg_circle.center.x)
        self.assertAlmostEqual(center["y"], fg_circle.center.y)
        self.assertAlmostEqual(radius, fg_circle.r)
        # Dict structure
        fg_circle_dict = fg_circle.to_dict()
        self.assertEqual(fg_circle_dict["type"], "SketchCircle")
        self.assertEqual(fg_circle_dict["construction_geom"], fg_circle.is_construction)
        center_uuid = fg_circle_dict["center_point"]
        center_uuid_point = self.point_map.uuid_map[center_uuid]
        self.assertAlmostEqual(center["x"], center_uuid_point.x)
        self.assertAlmostEqual(center["y"], center_uuid_point.y)
        self.assertAlmostEqual(fg_circle_dict["radius"], fg_circle.r)
