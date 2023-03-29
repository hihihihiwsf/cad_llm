import re
import numpy as np
from geometry.arc import Arc
from geometry.circle import Circle
from geometry.line import Line


def parse_string_to_curves(entities_string):
    entity_strings = [s.replace(" ", "") + ';' for s in entities_string.split(';') if s]
    curves = [parse_entity_string_to_curve(entity_string) for entity_string in entity_strings]
    return curves


def parse_entity_string_to_curve(entity_string):
    curve = None
    try:
        points = entity_string_to_points(entity_string)

        if len(points) == 2:
            curve = Line(points)
        elif len(points) == 3:
            curve = Arc(points)
        elif len(points) == 4:
            curve = Circle(points)
    except Exception as e:
        pass
    return curve


def entity_string_to_points(entity_string):
    flat_points = tuple(int(match.group()) for match in re.finditer(r'-?\d+', entity_string))
    points = [[flat_points[i], flat_points[i+1]] for i in range(0, len(flat_points), 2)]
    return np.array(points)
