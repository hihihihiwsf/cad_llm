from preprocess.deepmind_geometry.base import DeepmindBase
from matplotlib.patches import Circle


class DeepmindCircle(DeepmindBase):
    def __init__(self, dm_ent):
        super().__init__(dm_ent)
        self.name = "circle"
        try:
            self.center = (dm_ent["center"].get("x", 0), dm_ent["center"].get("y", 0))
            self.r = dm_ent["circleParams"].get("radius", 0)
        except Exception as e:
            self.exception = e

    def draw(self, ax, draw_points=False):
        circle = Circle(xy=self.center, radius=self.r, edgecolor="black", facecolor="none")
        ax.add_patch(circle)

    def to_points(self):
        x, y = self.center
        r = self.r

        return [(x, y - r), (x + r, y), (x, y + r), (x - r, y)]
