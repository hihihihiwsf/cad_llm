from .base import FusionGalleryBase
from .point import FusionGalleryPoint
from ..deepmind_geometry.base import DeepmindLine


class FusionGalleryLine(FusionGalleryBase):
    def __init__(self, ent, point_map):
        super().__init__(ent)
        # Currently we only support intialization from Deepmind data
        assert isinstance(ent, DeepmindLine)
        # Get the points referenced by the line
        self.start = FusionGalleryPoint.from_xy_map(ent.start[0], ent.start[1], point_map)
        self.end = FusionGalleryPoint.from_xy_map(ent.end[0], ent.end[1], point_map)

    def to_dict(self):
        """
        Make a Fusion 360 Gallery format dict like this
        {
            "type": "SketchLine",
            "construction_geom": false,
            "fixed": false,
            "fully_constrained": false,
            "reference": false,
            "visible": true,
            "start_point": "2bbb3a1c-e0c6-11ea-9072-c85b76a75ed8",
            "end_point": "2bbb3a1d-e0c6-11ea-a2d5-c85b76a75ed8"
        }
        """
        line_dict = self.create_common_entity_fields()
        line_dict.update(self.create_ent_points())
        line_dict["type"] = "SketchLine"
        return line_dict
