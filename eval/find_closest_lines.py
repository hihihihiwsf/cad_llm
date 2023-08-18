"""
Copied from SolidGenAutocomplete/blob/lambouj/thin_air_pick_network/data_proc/find_closest_lines.py

Project a list of points to the linear edges given.
Return the indices of the resulting lines.
If the edges have no lines at all then return an index
off the end of the array
"""

import numpy as np


def find_closest_point_on_line(
        start,
        end,
        datum
):
    # The line is parameterized with t = [0, 1]
    line_vec = end - start
    start_to_datum = datum - start
    len_seq = np.dot(line_vec, line_vec)
    if len_seq < 1e-7:
        # Avoid numerical error for very short segments
        t = 0.5
    else:
        t = np.dot(line_vec, start_to_datum) / len_seq
    if t < 0.0:
        return start
    if t > 1.0:
        return end
    return start + t * line_vec


def find_dist_to_edge(
        start,
        end,
        datum
):
    closest_point = find_closest_point_on_line(start, end, datum)
    return np.linalg.norm(closest_point - datum)


def find_dists_to_lines(
        vertices,
        edges,
        datum
):
    dists = []
    for edge in edges:
        if len(edge) == 2:
            start = vertices[edge[0]]
            end = vertices[edge[1]]
            dist = find_dist_to_edge(start, end, datum)
        else:
            dist = np.inf
        dists.append(dist)
    return np.array(dists)


def find_closest_line(
        vertices,
        edges,
        point_to_project
):
    assert isinstance(edges, list), "Edges must be a ragged list of int"
    dists = find_dists_to_lines(
        vertices,
        edges,
        point_to_project
    )
    if dists.min() < np.inf:
        return np.argmin(dists)
    return None


def find_closest_lines(
        vertices,
        edges,
        points_to_project
):
    assert isinstance(edges, list), "Edges must be a ragged list of int"
    closest_lines = []
    pad_index = -1
    for point in points_to_project:
        closest_line = find_closest_line(
            vertices,
            edges,
            point
        )
        if closest_line is not None:
            closest_lines.append(closest_line)
        else:
            closest_lines.append(pad_index)
    return np.array(closest_lines)
