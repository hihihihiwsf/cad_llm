import re
import numpy as np
from geometry.arc import Arc
from geometry.circle import Circle
from geometry.line import Line
from preprocess.preprocessing import sort_points

import sys 
sys.path.append("..") 
from dataset import sg_dataset

def new_get_point_entities(entities_string, sort=True):
    idx = 0
    token = [e.value for e in sg_dataset.Token]
    point_entity = []
    if len(entities_string)==0:
        return point_entity
    
    while(idx < len(entities_string)-1):
        val = entities_string[idx]
        if val in token:
            if val == sg_dataset.Token.Line and idx+4<len(entities_string)-1:
                point_entity.append(entities_string[idx+1:idx+4].numpy().tolist())
                idx = idx + 4
            elif val == sg_dataset.Token.Curve and idx+6<len(entities_string)-1:
                point_entity.append((entities_string[idx+1:idx+6]).numpy().tolist())
                idx = idx + 6
            elif val == sg_dataset.Token.Circle and idx+8<len(entities_string)-1:
                point_entity.append((entities_string[idx+1:idx+8]).numpy().tolist())
                idx = idx + 8
            else:
                idx += 1
        else:
            idx += 1
            
    return torch.tensor(point_entity)


def get_point_entities(entities_string, sort=True):
    """
    Convert a string describing multiple entities to a list of point entities
    Each point entity is a list of tuple points
    If sort is True, sort the points in each entity
    """
    entity_strings = [s.replace(" ", "") + ';' for s in entities_string.split(';') if s]
    #entity_strings = entities_string

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
    except Exception as e:
        pass
    return curve
