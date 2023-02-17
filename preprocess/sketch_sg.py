"""

Based on https://git.autodesk.com/Research/SketchDL/blob/master/datagen/sketch_sg.py
Modified to convert to a dictionary with obj file data instead of saving an obj file

"""

import math
import numpy as np
import importlib

from sketch_point import SketchPoint
import sketch_base
importlib.reload(sketch_base)
from sketch_base import SketchBase

from sketchgraphs.data._entity import Arc, Circle, Line


class SketchSG(SketchBase):
    """

    Class representing a Sketch from the 
    SketchGraphs Dataset

    """
    def __init__(self, sketch, sketch_name):
        super().__init__()
        self.sketch = sketch
        self.sketch_name = sketch_name

    def convert(self):
        """Convert to SketchDL obj format"""
        super().convert(None)

        curves = []
        # [ Line, Arc, Circle ]
        curve_type_counts = np.zeros(3, dtype=np.long)
        construction_count = 0
        for ent in self.sketch.entities.values():
            # Ignore construction lines
            if ent.isConstruction:
                construction_count += 1
                continue
            curve = None
            if isinstance(ent, Line):
                curve = self.convert_line(ent)
                curve_type_counts[0] += 1
            elif isinstance(ent, Arc):
                curve = self.convert_arc(ent)
                curve_type_counts[1] += 1
            elif isinstance(ent, Circle):
                curve = self.convert_circle(ent)
                curve_type_counts[2] += 1
            if curve:
                curves.append(curve)
        curve_count = np.sum(curve_type_counts)
        # Return if the sketch doesn't have any curves
        if curve_count == 0 or len(self.point_map) == 0:
            # If we have construction geometry in an empty sketch
            # We want to return None and not write an obj
            # as this is expected behaviour from SketchGraphs
            if construction_count != 0:
                return None

        vertices = np.array([(pt.x, pt.y) for pt in self.point_map.values()])
        return dict(name=self.sketch_name, vertices=vertices, curves=curves)

    def convert_line(self, line: Line):
        """Convert a single line"""
        start_x, start_y = line.start_point
        end_x, end_y = line.end_point
        start = self.get_point(start_x, start_y)
        end = self.get_point(end_x, end_y)
        self.add_edge(start.index, end.index)
        return [start.index, end.index]

    def convert_circle(self, circle: Circle):
        """Convert a single circle"""
        center = SketchPoint(circle.xCenter, circle.yCenter)
        top_raw, right_raw, bottom_raw, left_raw = self.get_circle_points(center, circle.radius)
        # Get points with an index
        top = self.get_point(top_raw.x, top_raw.y, weld=False)
        right = self.get_point(right_raw.x, right_raw.y, weld=False)
        bottom = self.get_point(bottom_raw.x, bottom_raw.y, weld=False)
        left = self.get_point(left_raw.x, left_raw.y, weld=False)
        self.add_edge(top.index, right.index)
        self.add_edge(right.index, bottom.index)
        self.add_edge(bottom.index, left.index)
        self.add_edge(left.index, top.index)
        return [top.index, right.index, bottom.index, left.index]

    def get_arc_mid_point(self, arc):
        """Get the mid point on an arc"""
        center = SketchPoint(arc.xCenter, arc.yCenter)
        ref_vec_angle = math.atan2(arc.yDir, arc.xDir)
        radius = arc.radius
        startParam = arc.startParam
        endParam = arc.endParam
        if arc.clockwise:
            startParam, endParam = -endParam, -startParam
        mid_angle = ref_vec_angle + (endParam + startParam) / 2.0
        mid_x = center.x + radius * math.cos(mid_angle)
        mid_y = center.y + radius * math.sin(mid_angle)
        return SketchPoint(mid_x, mid_y)

    def convert_arc(self, arc: Arc):
        """Convert a single arc"""
        start_x, start_y = arc.start_point
        end_x, end_y = arc.end_point
        # Get points with an index
        start = self.get_point(start_x, start_y)
        mid_raw = self.get_arc_mid_point(arc)
        mid = self.get_point(mid_raw.x, mid_raw.y, weld=False)
        end = self.get_point(end_x, end_y)
        self.add_edge(start.index, mid.index)
        self.add_edge(mid.index, end.index)
        return [start.index, mid.index, end.index]
