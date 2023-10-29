import re
import numpy as np
from geometry.arc import Arc
from geometry.circle import Circle
from geometry.line import Line
from preprocess.preprocessing import sort_points


def get_pair_constraints(entities_string, sort=True):
    constraint_strings = [s.replace(" ", "") + ';' for s in entities_string.split(';') if s]
    points_constraints = [sort_constraints(constraint) for constraint in constraint_strings]
    
    return points_constraints

def sort_constraints(constraint_string):
    l_len = [2,3]
    data = constraint_string[:-1].split(',')
    if len(data) not in l_len:
        return None
    e_items = [item for item in data if isinstance(item, str) and item.startswith('e')]
    numbers = [item for item in data if item.strip().isdigit()]
    if len(numbers)<1:
        return None
    if len(numbers)>1 or int(numbers[0])<65 or int(numbers[0])>77:
        return None
    sorted_parts = sorted(e_items) + numbers
    return ','.join(sorted_parts)+';'



def get_point_entities(entities_string, sort=True):
    """
    Convert a string describing multiple entities to a list of point entities
    Each point entity is a list of tuple points
    If sort is True, sort the points in each entity
    """
    entity_strings = [s.replace(" ", "") + ';' for s in entities_string.split(';') if s]
    point_entities = [get_point_entity(entity_string) for entity_string in entity_strings]
    if sort:
        point_entities = [sort_points(points) for points in point_entities]
    return point_entities



def get_point_entity(entity_string):
    """
    Convert a string describing a single entity to a list of tuple (x, y) points
    If the number of coordinates is odd the entity is invalid and None is returned
    """
    flat_points = tuple(int(match.group()) for match in re.finditer(r'-?\d+', entity_string))
    if len(flat_points) % 2 != 0:
        return None
    points = [(flat_points[i], flat_points[i+1]) for i in range(0, len(flat_points), 2)]
    return points


def get_curves(point_entities):
    """
    Convert point entities to curves
    Invalid entities may result in None curves or invalid curves (see flag curve.good)
    """
    return [get_curve(points) for points in point_entities]


def get_curve(points):
    """
    Convert a list of tuple points to curves
    Invalid entities may result in None curves or invalid curves (see flag curve.good)
    """
    if not points:
        return None

    curve = None
    try:
        points = np.array(points)
        if len(points) == 2:
            curve = Line(points)
        elif len(points) == 3:
            curve = Arc(points)
        elif len(points) == 4:
            curve = Circle(points)
        
        # if curve ==None:
        #     print("curve is None:")
        #     print(points)
    except Exception as e:
        pass
    return curve