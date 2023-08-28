import numpy as np

from geometry.parse import get_point_entity
from preprocess.preprocessing import normalize_and_quantize_vertices, sort_points


def process_for_syn_constraints(sketch, return_mid_points=False):
    vertices = np.array(sketch["vertices"])
    edges = sketch["edges"]

    if return_mid_points:
        # Calculate mid_point for each entity, None values for arcs and circles
        mid_points = [get_midpoint(vertices, edge) for edge in edges]
        # Add non None vertices
        np_mid_points = np.array([p for p in mid_points if p is not None])
        first_mid_point_v_index = len(vertices)
        if len(np_mid_points):
            vertices = np.vstack([vertices, np_mid_points])

    # quantize vertices
    vertices = normalize_and_quantize_vertices(vertices, n_bits=6)

    # shift coordinate range to start at 0
    vertices += 32

    # combine vertices and curves back to entities (lists of points)
    entities = [[list(vertices[i]) for i in edge] for edge in edges]
    # sort points in each entity
    entities = [sort_points(points) for points in entities]

    quantized_mid_points = None
    if return_mid_points:
        # get quantized mid points
        quantized_mid_points = []
        curr_v_index = first_mid_point_v_index
        for mid_point in mid_points:
            if mid_point is None:
                quantized_mid_points.append(None)
            else:
                quantized_mid_points.append(vertices[curr_v_index].tolist())
                curr_v_index += 1

    return {"entities": entities, "mid_points": quantized_mid_points}



def get_midpoint(vertices, edge):
    if len(edge) != 2:
        return None
    i, j = edge
    mid_point = (vertices[i] + vertices[j]) / 2
    return mid_point


def entity_index_list_to_string(indices):
    return "".join([f"<ent_{i}>" for i in indices])


def entity_index_list_from_string(indices_str):
    index_token_strings = [token_str for token_str in indices_str.split("<") if token_str and token_str.startswith("ent_")]
    index_token_strings = [token_str.strip("ent_").split(">")[0] for token_str in index_token_strings if token_str]
    indices = [int(index_string) for index_string in index_token_strings if index_string]
    return indices


def pp_constraints_to_string(constraints, mid_points):
    mid_point_strings = [(f"<{p[0]}><{p[1]}>" if p is not None else None) for p in mid_points]

    horizontal_str = "".join([mid_point_strings[i] for i in constraints["horizontal"]])
    vertical_str = "".join([mid_point_strings[i] for i in constraints["vertical"]])

    parallel_strings = ["".join([mid_point_strings[i] for i in group]) for group in constraints["parallel"]]
    parallel_str = "<parallel_sep>".join(parallel_strings)

    flat_perpendicular_indices = sum(constraints["perpendicular"], list())
    perpendicular_str = "".join([mid_point_strings[i] for i in flat_perpendicular_indices])

    constraints_str = "<constraint_sep>".join([horizontal_str, vertical_str, parallel_str, perpendicular_str])

    return constraints_str


def pp_constraints_from_string(constraints_str, mid_points):
    parts = constraints_str.split("<constraint_sep>")
    if len(parts) < 4:
        parts = parts + [""] * (4 - len(parts))
    if len(parts) > 4:
        parts = parts[:4]

    horizontal_str, vertical_str, parallel_str, perpendicular_str = parts
    parallel_group_strs = parallel_str.split("<parallel_sep>")

    horizontal = get_point_entity(horizontal_str) or []
    vertical = get_point_entity(vertical_str) or []
    parallel = [get_point_entity(parallel_group_str) or [] for parallel_group_str in parallel_group_strs]
    # perpendicular
    perpendicular_flat = get_point_entity(perpendicular_str) or []
    k = len(perpendicular_flat) if len(perpendicular_flat) % 2 == 0 else len(perpendicular_flat) - 1
    perpendicular = [perpendicular_flat[i:i+2] for i in range(0, k, 2)]

    # Convert points to closest mid_points
    horizontal = [get_closest_mid_point(point, mid_points) for point in horizontal]
    vertical = [get_closest_mid_point(point, mid_points) for point in vertical]
    parallel = [[get_closest_mid_point(point, mid_points) for point in points] for points in parallel]
    perpendicular = [[get_closest_mid_point(point, mid_points) for point in points] for points in perpendicular]

    # deduplicate and sort
    horizontal = sorted(set(horizontal))
    vertical = sorted(set(vertical))

    parallel = sorted(set([tuple(indices) for indices in parallel]))
    parallel = [list(group_points) for group_points in parallel]

    perpendicular = sorted(set([tuple(indices) for indices in perpendicular]))
    perpendicular = [[p1, p2] for p1, p2 in perpendicular]

    return {"horizontal": horizontal, "vertical": vertical, "parallel": parallel, "perpendicular": perpendicular}


def get_closest_mid_point(point, mid_points):
    point = np.array(point)
    dists = np.array([np.linalg.norm(point - x) if x is not None else np.inf for x in mid_points])
    if not dists.min() < np.inf:
        return None
    return int(np.argmin(dists))


def constraints_to_string(constraints):
    horizontal_str = entity_index_list_to_string(constraints["horizontal"])
    vertical_str = entity_index_list_to_string(constraints["vertical"])

    parallel_strings = [entity_index_list_to_string(indices) for indices in constraints["parallel"]]
    parallel_str = "<parallel_sep>".join(parallel_strings)

    flat_perpendicular_indices = sum(constraints["perpendicular"], list())
    perpendicular_str = entity_index_list_to_string(flat_perpendicular_indices)

    constraints_str = "<constraint_sep>".join([horizontal_str, vertical_str, parallel_str, perpendicular_str])

    return constraints_str


def constraints_to_string_schema2(constraints):
    horizontal_str = entity_index_list_to_string(constraints["horizontal"])
    vertical_str = entity_index_list_to_string(constraints["vertical"])

    parallel_strings = [entity_index_list_to_string(indices) for indices in constraints["parallel"]]
    parallel_str = "<parallel_sep>".join(parallel_strings)

    flat_perpendicular_indices = sum(constraints["perpendicular"], list())
    perpendicular_str = entity_index_list_to_string(flat_perpendicular_indices)

    constraints_str = "<horizontal>" + horizontal_str
    constraints_str += "<vertical>" + vertical_str
    constraints_str += "<parallel>" + parallel_str
    constraints_str += "<perpendicular>" + perpendicular_str

    return constraints_str


def constraints_from_string(constraints_str):
    parts = constraints_str.split("<constraint_sep>")
    if len(parts) < 4:
        parts = parts + [""] * (4 - len(parts))
    if len(parts) > 4:
        parts = parts[:4]

    horizontal_str, vertical_str, parallel_str, perpendicular_str = parts
    parallel_group_strs = parallel_str.split("<parallel_sep>") if parallel_str else []

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


def constraints_from_string_schema2(constraints_str):
    constraints_str = constraints_str.replace("<horizontal>", "")
    for schema2_token_string in ["<vertical>", "<parallel>", "<perpendicular>"]:
        constraints_str = constraints_str.replace(schema2_token_string, "<constraint_sep>")

    return constraints_from_string(constraints_str)


def constraints_to_sets(constraints):
    return {
        "horizontal": set(constraints["horizontal"]),
        "vertical": set(constraints["vertical"]),
        "parallel": set([tuple(indices) for indices in constraints["parallel"]]),
        "perpendicular": set([tuple(pair) for pair in constraints["perpendicular"]]),
    }
