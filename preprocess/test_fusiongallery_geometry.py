import unittest
from deepmind_geometry import DeepmindPoint
from fusiongallery_geometry import FusionGalleryPoint


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

    def test_point(self):
        dm_start = DeepmindPoint(self.line["lineEntity"]["start"])
        fg_start = FusionGalleryPoint(dm_start)
        self.assertIsNone(fg_start)
