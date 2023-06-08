from preprocess.deepmind_geometry.base import DeepmindBase


class DeepmindLine(DeepmindBase):
    def __init__(self, dm_ent):
        super().__init__(dm_ent)
        self.name = "line"
        try:
            self.start = (dm_ent["start"].get("x", 0), dm_ent["start"].get("y", 0))
            self.end = (dm_ent["end"].get("x", 0), dm_ent["end"].get("y", 0))
        except Exception as e:
            self.exception = e

    def to_points(self):
        return [self.start, self.end]

    def draw(self, ax):
        ax.plot([self.start[0], self.end[0]], [self.start[1], self.end[1]], color="black")
