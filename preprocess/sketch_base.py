"""

Copied from https://git.autodesk.com/Research/SketchDL/blob/master/datagen/sketch_base.py

"""

from collections import OrderedDict
import uuid
import networkx as nx

from preprocess.sketch_point import SketchPoint

class SketchBase:
    """

    Base class representing a Sketch

    """
    def __init__(self):
        # Tolerance to weld points together in decimal points
        self.tol = 8
        self.point_map = OrderedDict()
        self.graph = nx.Graph()
    
    def clear(self):
        """Clear the point map between profiles"""
        self.point_map = OrderedDict()
        self.graph = nx.Graph()

    def convert(self, output_dir):
        """Convert to SketchDL obj format"""
        self.output_dir = output_dir
    
    def write_obj(self, file, curve_strings, curve_count, vertex_strings):
        """Write an .obj file with the curves and verts"""
        with open(file, "w") as fh:
            fh.write("# WaveFront *.obj file\n")
            fh.write(f"# Vertices: {len(self.point_map)}\n")
            fh.write(f"# Curves: {curve_count}\n\n")
            fh.write(vertex_strings)
            fh.write("\n")
            fh.write(curve_strings)

    def convert_vertices(self, keep_invalid=False):
        """Convert all the vertices to .obj format"""
        if not keep_invalid:
            assert len(self.point_map) > 0
        vertex_strings = ""
        for pt in self.point_map.values():
            # e.g. v 0.123 0.234 0.345 1.0
            vertex_string = f"v {pt.x} {pt.y} {pt.z}\n"
            vertex_strings += vertex_string
        return vertex_strings

    def add_edge(self, a, b):
        """Add an edge in the sketch graph"""
        self.graph.add_edge(a, b)
    
    def get_loop_count(self):
        """Get a count of the number of loops in the sketch"""
        cycles = nx.cycle_basis(self.graph)
        return len(cycles)

    def get_point(self, x, y, weld=True):
        """Get a point stored in the point map"""
        if weld:
            # Create a unique key based on the xy location
            # to weld together common points
            # e.g. "x-2.000000y-1.000000"
            h_x = self.hash_string_from_double(x, self.tol)
            h_y = self.hash_string_from_double(y, self.tol)
            unique_key = f"x{h_x}y{h_y}"
            if unique_key not in self.point_map:
                pt = SketchPoint(x, y)
                # .obj indices start at 1
                pt.index = len(self.point_map) + 1
                self.point_map[unique_key] = pt
            return self.point_map[unique_key]
        else:
            # Don't weld the point as we have created it
            # and know it is unique
            pt = SketchPoint(x, y)
            # .obj indices start at 1
            pt.index = len(self.point_map) + 1
            # Use a uuid as the key
            self.point_map[uuid.uuid1()] = pt
            return pt

    def hash_string_from_double(self, value, decimal_places):
        value_str = f"{value:.{decimal_places}e}"
        value_str = value_str.replace(".", "d")
        value_str = value_str.replace("+", "a")
        value_str = value_str.replace("-", "b")
        mantissa_and_exponent = value_str.split("e")
        # Put the exponent first and mantissa second
        return mantissa_and_exponent[1] + mantissa_and_exponent[0] 
    
    def get_circle_points(self, center, radius):
        """Get 4 points around a circle"""
        top = SketchPoint(center.x, center.y - radius)
        right = SketchPoint(center.x + radius, center.y)
        bottom = SketchPoint(center.x, center.y + radius)
        left = SketchPoint(center.x - radius, center.y)
        return top, right, bottom, left