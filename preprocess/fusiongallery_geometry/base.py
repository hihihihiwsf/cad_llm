"""

FusionGalleryBase class extended by other geometry classes

"""

import uuid

from preprocess.deepmind_geometry import *

class FusionGalleryBase:
    def __init__(self, ent):
        """
        Intialize a FusionGalleryArc

        Args
            ent (DeepmindBase): Deepmind entity, can be one of:
                        DeepmindArc, DeepmindCircle, DeepmindLine, DeepmindPoint
        """        
        # Currently we only support intialization from Deepmind data
        if not isinstance(ent, (DeepmindArc, DeepmindCircle, DeepmindLine, DeepmindPoint)):
            raise Exception("Unsupported initialization data, expected Deepmind data")
        
        self.ent = ent
        self.is_construction = ent.is_construction
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