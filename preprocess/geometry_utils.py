import numpy as np


def normalize_vertices_scale(vertices):
    """Scale the vertices so that the long diagonal of the bounding box is one."""
    vert_min = vertices.min(axis=0)
    vert_max = vertices.max(axis=0)
    extents = vert_max - vert_min
    scale = np.sqrt(np.sum(extents ** 2))
    return vertices / scale


def center_vertices(vertices):
    """Translate the vertices so that bounding box is centered at zero."""
    vert_min = vertices.min(axis=0)
    vert_max = vertices.max(axis=0)
    vert_center = 0.5 * (vert_min + vert_max)
    return vertices - vert_center


def quantize_verts(verts, n_bits=8):
    """Convert vertices in [-1., 1.] to discrete values in [0, n_bits**2 - 1]."""
    min_range = -0.5
    max_range = 0.5
    range_quantize = 2 ** n_bits - 1
    verts_quantize = (verts - min_range) * range_quantize / (max_range - min_range)
    return verts_quantize.astype("int32")


def center_and_scale(vertices):
    """
    Convert vertices to values in [-0.5, 0.5]
    """
    vertices = center_vertices(vertices)
    scale = np.max(vertices.min(axis=0) - vertices.max(axis=0))
    return vertices / scale


def normalize_and_quantize_vertices(vertices, n_bits=6):
    """
    Convert vertices to discrete values in [-(n_bits-1)**2, (n_bits-1)**2 - 1].
    e.g. n_bits=6: [-32, 31]
    This was used in first experiment.
    """
    vertices = center_and_scale(vertices)
    quantize_range = 2 ** n_bits - 1
    vertices = (vertices * quantize_range).astype("int32")
    return vertices
