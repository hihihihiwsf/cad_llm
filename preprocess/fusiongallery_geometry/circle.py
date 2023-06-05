from .base import FusionGalleryBase
from .point import FusionGalleryPoint
from ..deepmind_geometry.base import DeepmindCircle


class FusionGalleryCircle(FusionGalleryBase):
    def __init__(self, ent, point_map):
        super().__init__(ent)
        # Currently we only support intialization from Deepmind data
        assert isinstance(ent, DeepmindCircle)
        # Get the point referenced by the circle
        self.center = FusionGalleryPoint.from_xy_map(ent.center[0], ent.center[1], point_map)
        self.r = ent.r

    def to_dict(self, point_map):
        """
        Make a Fusion 360 Gallery format dict like this
        {
            "type": "SketchCircle",
            "construction_geom": false,
            "fixed": false,
            "fully_constrained": false,
            "reference": false,
            "visible": true,
            "center_point": "3554cea2-e0c6-11ea-bf64-c85b76a75ed8",
            "radius": 1.118033988749895
        }
        """
        circle_dict = self.create_common_entity_fields()
        circle_dict["type"] = "SketchCircle"
        circle_dict["center_point"] = self.center.uuid
        circle_dict["radius"] = self.r
        return circle_dict
