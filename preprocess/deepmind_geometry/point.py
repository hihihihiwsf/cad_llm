from preprocess.deepmind_geometry.base import DeepmindBase


class DeepmindPoint(DeepmindBase):
    def __init__(self, dm_ent):
        super().__init__(dm_ent)
        self.name = "point"
        try:
            # Use get() here to handle when x or y is missing and equals 0
            if "point" in dm_ent:
                self.point = (dm_ent["point"].get("x", 0), dm_ent["point"].get("y", 0))
            else:
                # Also handle points inside of entities
                self.point = (dm_ent.get("x", 0), dm_ent.get("y", 0))
            # For convenience 
            self.x = self.point[0]
            self.y = self.point[1]
        except Exception as e:
            self.exception = e

    def draw(self, ax):
        ax.plot([self.point[0]], [self.point[1]], color="red", zorder=100, marker="o")
