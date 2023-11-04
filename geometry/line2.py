import cv2
import matplotlib.lines as lines
import numpy as np
import numpy.random as npr

from geometry.curve2 import Curve
from geometry.opencv_colors import CV2_COLORS


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

        self.get_ranges()
        self._get_chol()
        
    def get_ranges(self):
        self.update_x(self.points[0][0])
        self.update_y(self.points[0][1])
        self.update_x(self.points[1][0])
        self.update_y(self.points[1][1])

    def draw(self, ax, draw_points=True, linewidth=4, color="black"):
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

    def draw_np(self, np_image, draw_points=True, linewidth=4, color="blue", cell_size=4):
        """ Draw the line on a quantized grid with cell of size (cell_size, cell_size) """

        shifted_points = self.get_shifted_points(cell_size=cell_size)

        cv2.line(np_image, shifted_points[0], shifted_points[1], CV2_COLORS[color], thickness=linewidth)

        if draw_points:
            self.draw_points_np(np_image, cell_size)

        return np_image

    def draw_pil(self, img_draw, draw_points=True, linewidth=1, color="blue", transform=None):
        assert self.good, "The curve is not in the good state"

        points = self.points
        if transform:
            points = [(transform(x), transform(y)) for x, y in points]

        img_draw.line(xy=points, fill=color, width=linewidth)

        if draw_points:
            self.draw_points_pil(img_draw, transform=transform)

        return img_draw
    
    def hand_draw(self, ax, draw_points=True, linewidth=4, color="black"):
        pt0, pt1 = self.points

        linestyle = "-"

        start_x = self.points[0][0]
        start_y = self.points[0][1]
        end_x = self.points[1][0]
        end_y = self.points[1][1]
        
        length = np.sqrt((end_x-start_x)**2 + (end_y-start_y)**2)
        max_idx = int(np.floor((length / self.scale) * self.resolution))

        y = self.scale * self.cK[:max_idx, :max_idx] @ npr.randn(max_idx)
        x = self.x[:max_idx] * self.scale

        theta = np.arctan2(end_y-start_y, end_x-start_x)
        newx = start_x + x * np.cos(theta) - y * np.sin(theta)
        newy = start_y + y * np.cos(theta) + x * np.sin(theta)
        
        if draw_points:
            marker = '.'
        else:
            marker = None
        
        ax.plot(newx, newy, color, linewidth=linewidth, marker=marker, linestyle=linestyle)
        
        return newx, newy