import uuid

from ..deepmind_geometry.base import DeepmindBase

class FusionGalleryBase:
    def __init__(self, ent):
        # Currently we only support intialization from Deepmind data
        if not issubclass(ent, DeepmindBase):
            raise Exception("Unsupported initialization data, expected Deepmind data")
        
        self.ent = ent
        self.is_construction = ent.get("isConstruction", False)
        self.uuid = str(uuid.uuid1())

    def create_common_entity_fields(self):
        """
        Create common Fusion 360 Gallery format entity fields
        """
        return  {
            "construction_geom": self.is_construction,
            "fixed": False,
            "fully_constrained": False,
            "reference": False,
            "visible": True
        }

    def create_ent_points(self):
        """
        Create a dict with the start and end points
        """
        if self.start is None or self.end is None:
            return {}
        start_name = self.start.uuid
        end_name = self.end.uuid
        return {
            "start_point": start_name,
            "end_point": end_name
        }