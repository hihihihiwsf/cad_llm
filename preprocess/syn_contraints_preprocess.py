from preprocess.preprocessing import normalize_and_quantize_vertices, sort_points, get_entities_string


def get_entities_for_syn_constraints(sketch_dict, quantize_bits=6, new_tokens=True):
    vertices = sketch_dict["vertices"]
    curves = [[i-1 for i in edge] for edge in sketch_dict["edges"]]

    # quantize vertices
    vertices = normalize_and_quantize_vertices(vertices=vertices, n_bits=quantize_bits)

    # combine vertices and curves back to entities (lists of points)
    entities = [[list(vertices[i - 1]) for i in curve] for curve in curves]

    # sort points in each entity
    entities = [sort_points(points) for points in entities]

    # convert to tuples for deduplication
    entities = [tuple(tuple(point) for point in points) for points in entities]

    # do not deduplicate entities
    # deduped_entities = list(set(entities))
    # if len(deduped_entities) != len(entities):
    #     print("duplicate entities")

    # do not remove degenerate entities e.g. line with same start and end point
    # if any([len(points) != len(set(points)) for points in entities]):
    #     print("degenerate entity found in entities", entities)

    # do not sort entities

    # convert to strings
    entities_string = get_entities_string(entities, new_tokens=new_tokens)

    return entities_string


def entity_index_list_to_string(indices):
    return "".join([f"<ent_{i}>" for i in indices])


def entity_index_list_from_string(indices_str):
    index_strings = [part.strip(">") for part in indices_str.split("<ent_") if part]
    indices = [int(index_str) for index_str in index_strings]
    return indices


def constraints_to_string(constraints):
    horizontal_str = entity_index_list_to_string(constraints["horizontal"])
    vertical_str = entity_index_list_to_string(constraints["vertical"])

    parallel_strings = [entity_index_list_to_string(indices) for indices in constraints["parallel"]]
    parallel_str = "<parallel_sep>".join(parallel_strings)

    flat_perpendicular_indices = sum(constraints["perpendicular"], list())
    perpendicular_str = entity_index_list_to_string(flat_perpendicular_indices)

    constraints_str = "<constraint_sep>".join([horizontal_str, vertical_str, parallel_str, perpendicular_str])

    return constraints_str


def safe_constraints_sets_from_string(constraints_str):
    try:
        return constraints_sets_from_string(constraints_str)
    except Exception as e:
        return {"horizontal": set(), "vertical": set(), "parallel": set(), "perpendicular": set()}


def constraints_from_string(constraints_str):
    parts = constraints_str.split("<constraint_sep>")
    if len(parts) < 4:
        parts = parts + [None] * (4 - len(parts))
    if len(parts) > 4:
        parts = parts[:4]

    horizontal_str, vertical_str, parallel_str, perpendicular_str = parts

    horizontal = entity_index_list_from_string(horizontal_str)
    vertical = entity_index_list_from_string(vertical_str)

    # parallel
    parallel_strings = parallel_str.split("<parallel_sep>")
    parallel = [entity_index_list_from_string(parallel_str) for parallel_str in parallel_strings]

    # perpendicular
    perpendicular_flat = entity_index_list_from_string(perpendicular_str)
    if len(perpendicular_flat) % 2 != 0:
        perpendicular_flat = perpendicular_flat[:-1]
    perpendicular = [tuple(perpendicular_flat[i:i+2]) for i in range(0, len(perpendicular_flat), 2)]

    return {"horizontal": horizontal, "vertical": vertical, "parallel": parallel, "perpendicular": perpendicular}


def constraints_sets_from_string(constraints_str):
    constraints = constraints_from_string(constraints_str)

    constraints_sets = {}
    constraints_sets["horizontal"] = set(constraints["horizontal"])
    constraints_sets["vertical"] = set(constraints["vertical"])
    constraints_sets["parallel"] = set([set(indices) for indices in constraints["parallel"]])
    constraints_sets["perpendicular"] = set([tuple(pair) for pair in constraints["perpendicular"]])

    return constraints_sets
