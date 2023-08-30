"""

FusionGalleryPoint represents a sketch point in the Fusion 360 Gallery format

"""


from preprocess.fusiongallery_geometry.base import FusionGalleryBase
from preprocess.deepmind_geometry.point import DeepmindPoint


class FusionGalleryPoint(FusionGalleryBase):
    def __init__(self, ent):
        """
        Intialize a FusionGalleryPoint

        Args
            ent (DeepmindPoint): point
        """        
        super().__init__(ent)
        # Currently we only support intialization from Deepmind data
        assert isinstance(ent, DeepmindPoint)
        self.point = self.ent.point
        # For convenience 
        self.x = self.ent.point[0]
        self.y = self.ent.point[1]
        # Hash key used to merge points
        self.key = FusionGalleryPoint.get_key(self.x, self.y)
    
    @staticmethod
    def from_xy_map(x, y, point_map):
        """
        Return a FusionGalleryPoint object
        from a point map dictionary
        so we can reference to a canonical set of points
        """
        key = FusionGalleryPoint.get_key(x, y)
        if key in point_map:
            return point_map[key]
        return None

    @staticmethod
    def get_key(x, y, precision=15):
        """
        Return a hash key used to merge points
        """
        return f"{x:.{precision}f}_{y:.{precision}f}"

    def to_dict(self):
        """
        Make a Fusion 360 Gallery format dict like this
        {
            "type": "Point3D",
            "x": 0.0,
            "y": 0.0,
            "z": 0.0
        }
        """
        return {
            "type": "Point3D",
            "x": self.x,
            "y": self.y,
            "z": 0.0,
            "fixed": False,
            "fully_constrained": False,
            "linked": False,
            "reference": False,
            "visible": False            
        }
