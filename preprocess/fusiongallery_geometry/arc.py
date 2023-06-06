import math
import numpy as np

from .base import FusionGalleryBase
from .point import FusionGalleryPoint
from ..deepmind_geometry.base import DeepmindArc


class FusionGalleryArc(FusionGalleryBase):
    def __init__(self, ent, point_map):
        super().__init__(ent)
        # Currently we only support intialization from Deepmind data
        assert isinstance(ent, DeepmindArc)
        # Get the points referenced by the line
        self.center = FusionGalleryPoint.from_xy_map(ent.center[0], ent.center[1], point_map)
        self.start = FusionGalleryPoint.from_xy_map(ent.start[0], ent.start[1], point_map)
        self.end = FusionGalleryPoint.from_xy_map(ent.end[0], ent.end[1], point_map)
        self.is_clockwise = ent.is_clockwise

    def to_dict(self):
        """
        Make a Fusion 360 Gallery format dict like this
        {
            "type": "SketchArc",
            "construction_geom": false,
            "fixed": false,
            "fully_constrained": false,
            "reference": false,
            "visible": true,
            "start_point": "35d9b5d8-e0c6-11ea-8483-c85b76a75ed8",
            "end_point": "35d98eb8-e0c6-11ea-88b3-c85b76a75ed8",
            "center_point": "35da534a-e0c6-11ea-a2b4-c85b76a75ed8",
            "radius": 5.0,
            "reference_vector": {
                "type": "Vector3D",
                "x": -1.7763568394002506e-16,
                "y": 1.0,
                "z": 0.0,
                "length": 1.0
            },
            "start_angle": 0.0,
            "end_angle": 3.141592653589793
        },
        """
        arc_dict = self.create_common_entity_fields()
        arc_dict["type"] = "SketchArc"
        arc_dict.update(self.create_ent_points())
        if self.is_clockwise:
            # Fusion Gallery data uses the assumption that
            # an arc's direction is always anti-clockwise
            temp = arc_dict["start_point"]
            arc_dict["start_point"] = arc_dict["end_point"]
            arc_dict["end_point"] = temp
        arc_dict["center_point"] = self.center.uuid
        arc_dict["radius"] = self.get_radius()

        ref_vec = self.get_reference_vector()
        arc_dict["reference_vector"] = {
            "type": "Vector3D",
            "x": ref_vec[0],
            "y": ref_vec[1],
            "z": 0.0,
            "length": 1.0
        }

        angle_x_to_dir = FusionGalleryArc.angle_from_vector_to_x(ref_vec)
        center = self.merged_points[arc_dict["center_point"]]
        start_point = self.merged_points[arc_dict["start_point"]]
        end_point = self.merged_points[arc_dict["end_point"]]
        angle_x_to_start = self.angle_of_two_points_from_x(center, start_point)
        angle_x_to_end = self.angle_of_two_points_from_x(center, end_point)
        start_angle = angle_x_to_start - angle_x_to_dir
        end_angle = angle_x_to_end - angle_x_to_dir
        arc_dict["start_angle"] = start_angle
        arc_dict["end_angle"] = end_angle
        return arc_dict
    
    def get_radius(self):
        return math.sqrt((self.center.x - self.start.x) ** 2 + (self.center.y - self.start.y) ** 2)
    
    def get_reference_vector(self):
        """"Get the unitized vector from the center point to the start point"""
        return FusionGalleryArc.get_vector_between_pts(
            self.center.point, self.start.point
        )

    @staticmethod
    def get_vector_between_pts(pt1, pt2):
        """Get the unit vector between two points"""
        vec = pt2 - pt1
        length = np.linalg.norm(vec)
        if length > 1e-8:
            nvec = vec / length
        else:
            nvec = [1, 0]
        return nvec
    
    @staticmethod
    def angle_from_vector_to_x(vec):
        vec = [
           np.clip(vec[0], -1, 1),
           np.clip(vec[1], -1, 1)
        ]
        angle = 0.0
        # 2 | 1
        #-------
        # 3 | 4
        if vec[0] >=0:
            if vec[1] >= 0:
                # Qadrant 1
                angle = math.asin(vec[1])
            else:
                # Qadrant 4
                angle = 2.0*math.pi - math.asin(-vec[1])
        else:
            if vec[1] >= 0:
                # Qadrant 2
                angle = math.pi - math.asin(vec[1])
            else:
                # Qadrant 3
                angle = math.pi + math.asin(-vec[1])
        return angle


