import numpy as np
import matplotlib.patches as patches
from geometry.curve import Curve
import cv2
from geometry.opencv_colors import CV2_COLORS


class Circle(Curve):
    def __init__(self, points):
        assert len(points) == 4, "Circle must be defined by 4 points"
        super(Circle, self).__init__(points)
        self.find_circle_geometry()

    def draw(self, ax, draw_points=True, linewidth=1, color="red"):
        assert self.good, "The curve is not in the good state"
        ap = patches.Circle(self.center, self.radius, lw=linewidth, fill=None, color=color)
        ax.add_patch(ap)
        if draw_points:
            self.draw_points(ax)

    def draw_np(self, np_image, draw_points=True, linewidth=2, color="red", cell_size=4):
        """ Draw the line on a quantized grid with cell of size (cell_size, cell_size) """

        shifted_center = np.rint(self.shift_point(self.center, cell_size=cell_size)).astype(dtype=np.int32)
        radius = np.rint(self.radius * cell_size).astype(dtype=np.uint)

        cv2.circle(
            np_image,
            center=shifted_center,
            radius=radius,
            color=CV2_COLORS[color],
            thickness=linewidth,
        )

        if draw_points:
            self.draw_points_np(np_image, cell_size=cell_size)

        return np_image

    def find_circle_geometry(self):
        mid = np.mean(self.points, axis=0)

        # This algorithm comes from https://scipy-cookbook.readthedocs.io/items/Least_Squares_Circle.html
        # Calculation of the reduced coordinates
        reduced_points = self.points - mid
        u = reduced_points[:, 0]
        v = reduced_points[:, 1]

        try:        
            # linear system defining the center (uc, vc) in reduced coordinates:
            #    Suu * uc +  Suv * vc = (Suuu + Suvv)/2
            #    Suv * uc +  Svv * vc = (Suuv + Svvv)/2
            Suv  = np.dot(u, v)
            Suu  = np.dot(u, u)
            Svv  = np.dot(v, v)
            Suuv = (u*u*v).sum()
            Suvv = (u*v*v).sum()
            Suuu = (u*u*u).sum()
            Svvv = (v*v*v).sum()

            # Solving the linear system
            A = np.array([[Suu, Suv], [Suv, Svv]])
            B = np.array([Suuu + Suvv, Svvv + Suuv])/2.0
            c = np.linalg.solve(A, B)
        except:
            # This is the case where the points are coincident or colinear
            self.invalid_reason = "Circle has zero length"
            return 

        self.center = c + mid

        center_to_data = self.points - self.center
        self.radius = (np.linalg.norm(center_to_data, axis=1)).mean()

        # The circle parameters were found successfully 
        self.good = True
