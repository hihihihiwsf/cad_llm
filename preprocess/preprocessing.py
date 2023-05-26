import numpy as np


def preprocess_sketch(sketch_dict, quantize_bits, new_tokens=False):
    if not sketch_dict:
        return None

    name = sketch_dict["name"]
    vertices = sketch_dict["vertices"]
    curves = sketch_dict["curves"]

    # quantize vertices
    vertices = normalize_and_quantize_vertices(vertices=vertices, n_bits=quantize_bits)

    # combine vertices and curves back to entities (lists of points)
    entities = [[list(vertices[i - 1]) for i in curve] for curve in curves]

    # convert to tuples for deduplication
    entities = [tuple(tuple(point) for point in points) for points in entities]

    # remove degenerate entities e.g. line with same start and end point
    entities = [points for points in entities if len(points) == len(set(points))]

    # make a copy to leave in user order
    user_ordered_entities = [points for points in entities]

    # sort points in each entity, keep entity order
    sorted_entities = [sort_points(points) for points in entities]

    # find duplicate entities using sorted_entities as keys
    seen = set()
    deduped_ent_indices = set()
    for i, sorted_entity in enumerate(sorted_entities):
        if sorted_entity in seen:
            continue
        seen.add(sorted_entity)
        deduped_ent_indices.add(i)

    # filter out sketches with only none or one entity remaining
    if len(deduped_ent_indices) <= 1:
        return None

    # deduplicate entities
    user_ordered_entities = [points for i, points in enumerate(user_ordered_entities) if deduped_ent_indices]
    sorted_entities = [points for i, points in enumerate(sorted_entities) if i in deduped_ent_indices]

    # sort entities
    sorted_entities.sort()

    # convert to strings
    entities_string = get_entities_string(sorted_entities, new_tokens=new_tokens)
    user_ordered_entities_string = get_entities_string(user_ordered_entities, new_tokens=new_tokens)

    return dict(name=name, entities=entities_string, user_ordered_entities=user_ordered_entities_string)


def get_entities_string(entities, new_tokens):
    # flatten [[x1, y1], [x2, y2], ...] -> [x1, y1, x2, y2, ...]
    flat_entities = [sum(points, tuple()) for points in entities]
    # convert each entity to a string
    str_entities = []
    if not new_tokens:
        for ent in entities:
            all_points = []
            for p in ent:
                all_points.append(" ".join(str(x) for x in p))
            str_entities.append(",".join(p for p in all_points) + ";")

            
        # str_entities = [",".join([str(x) for x in ent]) + ";" for ent in flat_entities]
    else:
        str_entities = ["".join([f"<{x}>" for x in ent]) + ";" for ent in flat_entities]
    return str_entities


def sort_points(points):
    if not points:
        return None

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
    return tuple(points)


def center_vertices(vertices):
    """Translate the vertices so that bounding box is centered at zero."""
    vert_min = vertices.min(axis=0)
    vert_max = vertices.max(axis=0)
    vert_center = 0.5 * (vert_min + vert_max)
    return vertices - vert_center


def center_and_scale(vertices):
    """
    Convert vertices to values in [-0.5, 0.5]
    """
    vertices = center_vertices(vertices)
    scale = np.max(vertices.max(axis=0) - vertices.min(axis=0))
    return vertices / scale


def normalize_and_quantize_vertices(vertices, n_bits=6):
    """
    Convert vertices to discrete values in [-(n_bits-1)**2, (n_bits-1)**2 - 1].
    e.g. n_bits=6: [-32, 31]
    """
    vertices = center_and_scale(vertices)
    quantize_range = 2 ** n_bits - 1
    vertices = (vertices * quantize_range).astype("int32")
    return vertices
