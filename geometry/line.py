import matplotlib.lines as lines
import numpy as np

from geometry.curve import Curve

class Line(Curve):
    def __init__(self, points):
        assert len(points) == 2, "Line must be defined by two points"
        super(Line, self).__init__(points)
        pt0, pt1 = points

        tiny_tol = 1e-7
        if np.linalg.norm(pt1-pt0) > tiny_tol:
            # Check the length of the line is non-zero
            self.good = True
        else:
            self.invalid_reason = "Line has zero length"


    def draw(self, ax, draw_points=True, linewidth=1, color="black"):
        pt0, pt1 = self.points

        linestyle = "-"
        xdata = [pt0[0], pt1[0]]
        ydata = [pt0[1], pt1[1]]
        l1 = lines.Line2D(
            xdata, 
            ydata, 
            lw=linewidth, 
            linestyle=linestyle, 
            color=color, 
            axes=ax
        )
        ax.add_line(l1)
        if draw_points:
            self.draw_points(ax)
