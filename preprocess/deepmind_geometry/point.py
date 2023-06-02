from .base import DeepmindBase


class DeepmindPoint(DeepmindBase):
    def __init__(self, dm_ent):
        super().__init__(dm_ent)
        self.name = "point"
        try:
            if "point" in dm_ent:
                self.point = (dm_ent["point"].get("x", 0), dm_ent["point"].get("y", 0))
            else:
                # Also handle points inside of entities
                self.point = (dm_ent.get("x", 0), dm_ent.get("y", 0))
        except Exception as e:
            self.exception = e
        # Hash key used to merge points
        precision = 17
        self.key = f"{self.point[0]:.{precision}f}_{self.point[1]:.{precision}f}"

    def draw(self, ax):
        ax.plot([self.point[0]], [self.point[1]], color="red", zorder=100, marker="o")

    def to_fg_dict(self):
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
            "x": self.point[0],
            "y": self.point[1],
            "z": 0.0
        }