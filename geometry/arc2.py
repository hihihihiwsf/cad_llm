import cv2
import matplotlib.lines as lines
import matplotlib.patches as patches
import numpy as np
import numpy.random as npr

import math
import geometry.geom_utils as geom_utils
from geometry.curve2 import Curve
from geometry.line import Line
from geometry.opencv_colors import CV2_COLORS


class Arc(Curve):
    def __init__(self, points):
        assert len(points) == 3, "Arc must be defined by 3 points"
        super(Arc, self).__init__(points)
        
        #self.start_point=None
        self.find_arc_geometry()
        
        if self.good==True:
            self.get_ranges()
            self._get_chol()
        
    def get_ranges(self):
        self.update_x(self.start_point[0])
        self.update_y(self.start_point[1])
        self.update_x(self.end_point[0])
        self.update_y(self.end_point[1])


    def draw(self, ax, draw_points=True, linewidth=2, color="green"):
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

    #color should be green
    def hand_draw(self, ax, draw_points=True, linewidth=4, color="black"):  
        if not self.good:
            xdata, ydata = zip(self.points[0], self.points[2])
            line = Line((self.points[0],self.points[2]))
            line.hand_draw(ax, draw_points, linewidth,color)
            
            return 
        
        start_angle = geom_utils.rads_to_degs(self.start_angle_rads)
        end_angle = geom_utils.rads_to_degs(self.end_angle_rads)
        
        # angle = math.atan2(self.center[1], self.center[0]) * 180 / math.pi
        # start = start_angle * 180 / math.pi + angle
        # end = end_angle * 180 / math.pi + angle
        
        start = np.pi * start_angle / 360
        end = np.pi * end_angle / 360
        

        # if end < start:
        #     end += 2 * np.pi

        length = np.abs(self.radius * (end-start))
        try:
            max_idx = np.minimum(np.maximum(int(np.floor((length / np.maximum(self.scale,1e-6)) * self.resolution)), 1), 500)
            y = self.scale * self.cK[:max_idx, :max_idx] @ npr.randn(max_idx)
        except:
            import pdb;pdb.set_trace()
            
        thetas = np.linspace(start, end, max_idx)
        newx = self.center[0]+ (self.radius + y) * np.cos(thetas)
        newy = self.center[1] + (self.radius + y) * np.sin(thetas)
        
        ax.plot(newx, newy, color=color, linewidth=linewidth)
        if draw_points:
            self.draw_points(ax)
    
    def draw_np(self, np_image, draw_points=True, linewidth=2, color="green", cell_size=4):
        """ Draw the line on a quantized grid with cell of size (cell_size, cell_size) """

        shifted_points = self.get_shifted_points(cell_size=cell_size)

        if not self.good:
            # The points are co-linear, the arc is a line (probably due to quantization)
            cv2.line(np_image, shifted_points[0], shifted_points[1], CV2_COLORS[color], thickness=linewidth)

            if draw_points:
                self.draw_points_np(np_image, cell_size)

            return

        # Round to integers for plotting with cv2.ellipse
        center = np.rint(self.shift_point(self.center, cell_size=cell_size)).astype(np.int32)
        radius = np.rint(self.radius * cell_size).astype(np.int32)

        start_angle = geom_utils.rads_to_degs(self.start_angle_rads)
        end_angle = geom_utils.rads_to_degs(self.end_angle_rads)

        # workaround for what seems like a bug in opencv
        if end_angle < start_angle:
            end_angle += 360

        cv2.ellipse(np_image, center=center, axes=(radius, radius), angle=0, startAngle=start_angle, endAngle=end_angle,
                    color=CV2_COLORS[color], thickness=linewidth)

        if draw_points:
            self.draw_points_np(np_image, cell_size)

        return np_image

    def draw_pil(self, img_draw, draw_points=True, linewidth=1, color="green", transform=None):
        assert self.good, "The curve is not in the good state"

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
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        

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