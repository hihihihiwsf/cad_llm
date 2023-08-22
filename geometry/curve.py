import math
import cv2
from geometry.opencv_colors import CV2_COLORS


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

    def draw_points_np(self, np_image, cell_size, radius=2):
        """
        Draw markers for the points using opencv
        """
        for point in self.get_shifted_points(cell_size=cell_size):
            cv2.circle(np_image, point, radius=radius, color=CV2_COLORS["black"], thickness=-1)
    def draw_points_pil(self, img_draw, color="black", transform=None):
        r = 3
        points = self.points
        if transform:
            points = [(transform(x), transform(y)) for x, y in points]

        for x, y in points:
            img_draw.ellipse(xy=(x-r, y-r, x+r, y+r), fill=color, outline=None, width=1)

    def get_shifted_points(self, cell_size):
        """
        Shift points to center of cell in quantized grid
        """
        return cell_size * self.points + (cell_size // 2, cell_size // 2)

    def shift_point(self, point, cell_size):
        return cell_size * point + (cell_size // 2, cell_size // 2)
