from .base import DeepmindBase


class DeepmindPoint(DeepmindBase):
    def __init__(self, dm_ent):
        super().__init__(dm_ent)
        self.name = "point"
        try:
            self.point = (dm_ent["point"].get("x", 0), dm_ent["point"].get("y", 0))
        except Exception as e:
            self.exception = e

    def draw(self, ax):
        ax.plot([self.point[0]], [self.point[1]], color="red", zorder=100, marker="o")
