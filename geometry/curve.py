import math


class Curve:
    def __init__(self, points):
        self.points = points
        self.good = False
        self.invalid_reason = None

    def draw_points(self, ax):
        """
        Draw markers for the points
        """
        assert self.good, "The curve is not in the good state"
        for x, y in self.points:
            ax.plot(x, y, 'b.')
