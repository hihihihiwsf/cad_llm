import numpy as np

from geometry.parse import get_point_entity
from preprocess.preprocessing import normalize_and_quantize_vertices, sort_points, get_entities_string


def get_entities_for_syn_constraints(sketch_dict):
    vertices = sketch_dict["vertices"]
    curves = [[i-1 for i in edge] for edge in sketch_dict["edges"]]

    # quantize vertices
    vertices = normalize_and_quantize_vertices(vertices=vertices, n_bits=6)
    # shift coordinate range to start at 0
    vertices += 32

    # combine vertices and curves back to entities (lists of points)
    entities = [[list(vertices[i - 1]) for i in curve] for curve in curves]

    # sort points in each entity
    entities = [sort_points(points) for points in entities]

    # convert to tuples for deduplication
    entities = [tuple(tuple(point) for point in points) for points in entities]

    # do not deduplicate entities
    # do not remove degenerate entities e.g. line with same start and end point
    # do not sort entities

    # convert to strings
    entities_string = get_entities_string(entities, new_tokens=True)

    return entities_string


def entity_index_list_to_string(indices):
    return "".join([f"<ent_{i}>" for i in indices])


def entity_index_list_from_string(indices_str):
    index_strings = indices_str.split("<ent_")
    index_strings = [index_string.strip("> ") for index_string in index_strings]
    indices = [int(index_string) for index_string in index_strings if index_string]
    return indices


def get_pp_constraints_string(sketch):
    constraints = sketch["constraints"]
    vertices = sketch["vertices"]
    edges = sketch["edges"]
    mid_point_strings = get_midpoint_strings(vertices, edges)

    horizontal_str = "".join([mid_point_strings[i] for i in constraints["horizontal"]])
    vertical_str = "".join([mid_point_strings[i] for i in constraints["vertical"]])

    parallel_strings = ["".join([mid_point_strings[i] for i in group]) for group in constraints["parallel"]]
    parallel_str = "<parallel_sep>".join(parallel_strings)

    flat_perpendicular_indices = sum(constraints["perpendicular"], list())
    perpendicular_str = "".join([mid_point_strings[i] for i in flat_perpendicular_indices])

    constraints_str = "<constraint_sep>".join([horizontal_str, vertical_str, parallel_str, perpendicular_str])

    return constraints_str


def get_midpoint_strings(vertices, edges, quantize_bits=6):
    vertices = np.array(vertices)

    # Calculate mid-points
    mid_points = [get_midpoint(vertices, edge) for edge in edges]

    # Quantize mid-points according to entire sketch
    np_mid_points = np.array([p for p in mid_points if p is not None])

    if not len(np_mid_points):
        return [None] * len(edges)

    all_vertices = np.vstack([vertices, np_mid_points])
    all_vertices = normalize_and_quantize_vertices(vertices=all_vertices, n_bits=quantize_bits)
    # shift coordinate range to start at 0
    # all_vertices += 32

    # Convert to mid-points to strings
    mid_point_strings = []
    current_mid_point_v_index = len(vertices)
    for edge_index, mid_point in enumerate(mid_points):
        mid_point_string = None
        if mid_point is not None:
            x, y = all_vertices[current_mid_point_v_index]
            mid_point_string = f"<{x}><{y}>"
            current_mid_point_v_index += 1

        mid_point_strings.append(mid_point_string)

    return mid_point_strings


def get_midpoint(vertices, edge):
    if len(edge) != 2:
        return None
    i, j = edge
    mid_point = (vertices[i] + vertices[j]) / 2
    return mid_point


def pp_constraints_from_string(constraints_str):
    parts = constraints_str.split("<constraint_sep>")
    if len(parts) < 4:
        parts = parts + [""] * (4 - len(parts))
    if len(parts) > 4:
        parts = parts[:4]

    horizontal_str, vertical_str, parallel_str, perpendicular_str = parts
    parallel_group_strs = parallel_str.split("<parallel_sep>")

    horizontal = get_point_entity(horizontal_str) or []
    horizontal = sorted(set(horizontal))

    vertical = get_point_entity(vertical_str) or []
    vertical = sorted(set(vertical))

    # parallel
    parallel = []
    for parallel_group_str in parallel_group_strs:
        parallel_group = get_point_entity(parallel_group_str) or []
        parallel.append(sorted(set(parallel_group)))

    # deduplicate and sort
    parallel = sorted(set([tuple(group_points) for group_points in parallel]))
    # convert to list of lists
    parallel = [list(indices) for indices in parallel]

    # perpendicular
    # parser to pair tuples, deduplicate pairs, and sort
    perpendicular_flat = get_point_entity(perpendicular_str) or []
    if len(perpendicular_flat) % 2 != 0:
        perpendicular_flat = perpendicular_flat[:-1]
    perpendicular = [tuple(perpendicular_flat[i:i+2]) for i in range(0, len(perpendicular_flat), 2)]

    # deduplicate and sort
    perpendicular = sorted(set(perpendicular))
    # convert to list of lists
    perpendicular = [[p1, p2] for p1, p2 in perpendicular]

    return {"horizontal": horizontal, "vertical": vertical, "parallel": parallel, "perpendicular": perpendicular}


def safe_pp_constraints_from_string(constraints_str):
    try:
        return pp_constraints_from_string(constraints_str)
    except Exception as e:
        return {"horizontal": [], "vertical": [], "parallel": [], "perpendicular": []}


def constraints_to_string(constraints):
    horizontal_str = entity_index_list_to_string(constraints["horizontal"])
    vertical_str = entity_index_list_to_string(constraints["vertical"])

    parallel_strings = [entity_index_list_to_string(indices) for indices in constraints["parallel"]]
    parallel_str = "<parallel_sep>".join(parallel_strings)

    flat_perpendicular_indices = sum(constraints["perpendicular"], list())
    perpendicular_str = entity_index_list_to_string(flat_perpendicular_indices)

    constraints_str = "<constraint_sep>".join([horizontal_str, vertical_str, parallel_str, perpendicular_str])

    return constraints_str


def constraints_from_string(constraints_str):
    parts = constraints_str.split("<constraint_sep>")
    if len(parts) < 4:
        parts = parts + [None] * (4 - len(parts))
    if len(parts) > 4:
        parts = parts[:4]

    horizontal_str, vertical_str, parallel_str, perpendicular_str = parts
    parallel_group_strs = parallel_str.split("<parallel_sep>")

    horizontal = sorted(set(entity_index_list_from_string(horizontal_str)))
    vertical = sorted(set(entity_index_list_from_string(vertical_str)))

    # parallel
    parallel = []
    for parallel_group_str in parallel_group_strs:
        parallel.append(sorted(set(entity_index_list_from_string(parallel_group_str))))

    # deduplicate and sort
    parallel = sorted(set([tuple(group_indices) for group_indices in parallel]))
    # convert to list of lists
    parallel = [list(indices) for indices in parallel]

    # perpendicular
    # parser to pair tuples, deduplicate pairs, and sort
    perpendicular_flat = entity_index_list_from_string(perpendicular_str)
    if len(perpendicular_flat) % 2 != 0:
        perpendicular_flat = perpendicular_flat[:-1]
    perpendicular = [tuple(perpendicular_flat[i:i+2]) for i in range(0, len(perpendicular_flat), 2)]

    # deduplicate and sort
    perpendicular = sorted(set(perpendicular))
    # convert to list of lists
    perpendicular = [[i, j] for i, j in perpendicular]

    return {"horizontal": horizontal, "vertical": vertical, "parallel": parallel, "perpendicular": perpendicular}


def safe_constraints_from_string(constraints_str):
    try:
        return constraints_from_string(constraints_str)
    except Exception as e:
        return {"horizontal": [], "vertical": [], "parallel": [], "perpendicular": []}


def constraints_to_sets(constraints):
    return {
        "horizontal": set(constraints["horizontal"]),
        "vertical": set(constraints["vertical"]),
        "parallel": set([tuple(indices) for indices in constraints["parallel"]]),
        "perpendicular": set([tuple(pair) for pair in constraints["perpendicular"]]),
    }

