from preprocess.deepmind_geometry.base import DeepmindBase
import math
from matplotlib.patches import Arc


class DeepmindArc(DeepmindBase):
    def __init__(self, dm_ent):
        super().__init__(dm_ent)
        self.name = "arc"
        try:
            params = dm_ent["arcParams"]

            self.center = (dm_ent["center"].get("x", 0), dm_ent["center"].get("y", 0))
            self.start = (params["start"].get("x", 0), params["start"].get("y", 0))
            self.end = (params["end"].get("x", 0), params["end"].get("y", 0))
            self.is_clockwise = params.get("isClockwise", False)

        except Exception as e:
            self.exception = e
            raise e

    def get_mid(self):
        start, end = self.start, self.end
        if self.is_clockwise:
            start, end = end, start

        r = self.get_radius()
        angle1 = math.atan2(start[1] - self.center[1], start[0] - self.center[0]) % (2 * math.pi)
        angle2 = math.atan2(end[1] - self.center[1], end[0] - self.center[0]) % (2 * math.pi)
        if angle2 < angle1:
            angle2 += 2 * math.pi

        mid_angel = (angle1 + angle2) / 2
        mid = (self.center[0] + r * math.cos(mid_angel), self.center[1] + r * math.sin(mid_angel))
        return mid

    def get_radius(self):
        return math.sqrt((self.center[0] - self.start[0]) ** 2 + (self.center[1] - self.start[1]) ** 2)

    def to_points(self):
        return [self.start, self.get_mid(), self.end]

    def draw(self, ax, draw_points=False):
        center = self.center
        start = self.start
        end = self.end
        mid = self.get_mid()
        r = self.get_radius()

        if self.is_clockwise:
            start, end = end, start

        angle1 = 180 / math.pi * math.atan2(start[1] - center[1], start[0] - center[0])
        angle2 = 180 / math.pi * math.atan2(end[1] - center[1], end[0] - center[0])

        arc = Arc(center, 2 * r, 2 * r, angle=0, theta1=angle1, theta2=angle2,
                  edgecolor='black', facecolor='none')
        ax.add_patch(arc)

        if draw_points:
            ax.plot([start[0]], [start[1]], color="red", zorder=100, marker="o")
            ax.plot([mid[0]], [mid[1]], color="yellow", zorder=100, marker="o")
            ax.plot([end[0]], [end[1]], color="blue", zorder=100, marker="o")
