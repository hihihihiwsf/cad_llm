import cv2
import matplotlib.lines as lines
import matplotlib.patches as patches
import numpy as np

import geometry.geom_utils as geom_utils
from geometry.curve import Curve
from geometry.opencv_colors import CV2_COLORS


class Arc(Curve):
    def __init__(self, points):
        assert len(points) == 3, "Arc must be defined by 3 points"
        super(Arc, self).__init__(points)
        self.find_arc_geometry()

    def draw(self, ax, draw_points=True, linewidth=1, color="green"):
        if not self.good:
            # The points are co-linear, the arc is a line (probably due to quantization)
            xdata, ydata = zip(self.points[0], self.points[2])
            l1 = lines.Line2D(xdata, ydata, lw=linewidth, linestyle="-", color=color, axes=ax)
            ax.add_line(l1)

            if draw_points:
                self.draw_points(ax)
            return

        diameter = 2.0*self.radius
        start_angle = geom_utils.rads_to_degs(self.start_angle_rads)
        end_angle = geom_utils.rads_to_degs(self.end_angle_rads)
        ap = patches.Arc(
            self.center, 
            diameter,
            diameter,
            angle=0, 
            theta1=start_angle, 
            theta2=end_angle,
            color=color,
            fc="none",
            lw=linewidth
        )
        ax.add_patch(ap)
        if draw_points:
            self.draw_points(ax)

    def draw_np(self, np_image, draw_points=True, linewidth=1, color="green", cell_size=4):
        """ Draw the line on a quantized grid with cell of size (cell_size, cell_size) """

        shifted_points = self.get_shifted_points(cell_size=cell_size)

        if not self.good:
            # The points are co-linear, the arc is a line (probably due to quantization)
            cv2.line(np_image, shifted_points[0], shifted_points[1], CV2_COLORS[color], thickness=linewidth)

            if draw_points:
                self.draw_points_np(np_image, cell_size)

            return

        center = self.shift_point(self.center, cell_size=cell_size).astype(dtype=np.int)
        radius = (self.radius * cell_size).astype(dtype=np.uint)

        start_angle = geom_utils.rads_to_degs(self.start_angle_rads)
        end_angle = geom_utils.rads_to_degs(self.end_angle_rads)

        # workaround for what seems like a bug in opencv
        if end_angle < start_angle:
            end_angle += 360

        try:
            cv2.ellipse(np_image, center=center, axes=(radius, radius), angle=0, startAngle=start_angle, endAngle=end_angle,
                        color=CV2_COLORS[color], thickness=linewidth)
        except:
            print('stop')

        if draw_points:
            self.draw_points_np(np_image, cell_size)

        return np_image


    def draw_pil(self, img_draw, draw_points=True, linewidth=1, color="green", transform=None):
        if not self.good:
            points = self.points
            if transform:
                points = [(transform(x), transform(y)) for x, y in points]

                img_draw.line(xy=points, fill="blue", width=linewidth)

            if draw_points:
                self.draw_points_pil(img_draw, transform=transform)
            
            return img_draw
            
        # assert self.good, "The curve is not in the good state"

        lower_left = self.center - self.radius
        upper_right = self.center + self.radius

        bounding_box_points = [(lower_left[0], lower_left[1]), (upper_right[0], upper_right[1])]
        if transform:
            bounding_box_points = [(transform(x), transform(y)) for x, y in bounding_box_points]

        start_angle = geom_utils.rads_to_degs(self.start_angle_rads)
        end_angle = geom_utils.rads_to_degs(self.end_angle_rads)
        img_draw.arc(bounding_box_points, start=start_angle, end=end_angle, fill=color, width=linewidth)

        if draw_points:
            self.draw_points_pil(img_draw=img_draw, transform=transform)

        return img_draw
    

    def find_arc_geometry(self):
        #     Subject 1.04: How do I generate a circle through three points?

        # Let the three given points be a, b, c.  Use _0 and _1 to represent
        # x and y coordinates. The coordinates of the center p=(p_0,p_1) of
        # the circle determined by a, b, and c are:

        #     A = b_0 - a_0;
        #     B = b_1 - a_1;
        #     C = c_0 - a_0;
        #     D = c_1 - a_1;
        
        #     E = A*(a_0 + b_0) + B*(a_1 + b_1);
        #     F = C*(a_0 + c_0) + D*(a_1 + c_1);
        
        #     G = 2.0*(A*(c_1 - b_1)-B*(c_0 - b_0));
        
        #     p_0 = (D*E - B*F) / G;
        #     p_1 = (A*F - C*E) / G;
    
        # If G is zero then the three points are collinear and no finite-radius
        # circle through them exists.  Otherwise, the radius of the circle is:

        #         r^2 = (a_0 - p_0)^2 + (a_1 - p_1)^2

        # Reference:

        # [O' Rourke (C)] p. 201. Simplified by Jim Ward.
        a, b, c = self.points

        A = b[0] - a[0] 
        B = b[1] - a[1]
        C = c[0] - a[0]
        D = c[1] - a[1]
    
        E = A*(a[0] + b[0]) + B*(a[1] + b[1])
        F = C*(a[0] + c[0]) + D*(a[1] + c[1])
    
        G = 2.0*(A*(c[1] - b[1])-B*(c[0] - b[0]))

        tiny_tol = 1e-7

        if np.abs(G) < tiny_tol:
            self.invalid_reason = "Arc is a line or has zero length"
            return
    
        p_0 = (D*E - B*F) / G
        p_1 = (A*F - C*E) / G

        self.center = np.array([p_0, p_1])
        self.radius = np.linalg.norm(self.center - a)

        angles = []
        for point in self.points:
            angle = geom_utils.angle_from_vector_to_x(point - self.center)
            angles.append(angle)

        ab = b-a
        ac = c-a
        cp = np.cross(ab, ac)
        if cp >= 0:
            self.start_angle_rads = angles[0]
            self.end_angle_rads = angles[2]
            self.start_point = a
            self.end_point = c
        else:
            self.start_angle_rads = angles[2]
            self.end_angle_rads = angles[0]
            self.start_point = c
            self.end_point = a

        # The arc parameters were found successfully 
        self.good = True