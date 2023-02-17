from preprocess.geometry_utils import normalize_and_quantize_vertices


def preprocess_sketch(sketch_dict, quantize_bits, new_tokens=False):
    if not sketch_dict:
        return None

    name = sketch_dict["name"]
    vertices = sketch_dict["vertices"]
    curves = sketch_dict["curves"]

    # filter out sketches with only 1 curve
    if len(curves) == 1:
        return None

    # quantize vertices
    vertices = normalize_and_quantize_vertices(vertices=vertices, n_bits=quantize_bits)
    # combine vertices and curves back to entities (lists of points)
    entities = [[list(vertices[i - 1]) for i in curve] for curve in curves]
    # sort points in each entity
    entities = [sort_points(points) for points in entities]
    # sort entities
    entities = sorted(entities)
    # flatten [[x1, y1], [x2, y2], ...] -> [x1, y1, x2, y2, ...]
    flat_entities = [sum(points, []) for points in entities]
    # convert each entity to a string
    if not new_tokens:
        str_entities = [",".join([str(x) for x in ent]) + ";" for ent in flat_entities]
    else:
        str_entities = ["".join([f"<{x}>" for x in ent]) + ";" for ent in flat_entities]

    return dict(name=name, entities=str_entities)


def sort_points(points):
    if len(points) == 2:  # Line
        points = sorted(points)
    elif len(points) == 3:  # Arc
        start, mid, end = points
        if start > end:
            points = [end, mid, start]
    if len(points) == 4:  # Circle
        # top, right, bottom, left = points
        # sort -> left, top, bottom, right
        points = sorted(points)
    return points
