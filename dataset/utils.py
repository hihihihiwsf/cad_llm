import random
import math


def get_quantized(vertices, n_bits):
    """
    Convert normalized vertices to discrete values in [-(n_bits-1)**2, (n_bits-1)**2 - 1].
    e.g. n_bits=6: [-32, 31]
    This was used in first experiment.
    """
    quantize_range = 2 ** n_bits - 1
    return (vertices * quantize_range).astype("int32")


def choose_random_subset(n, subset_range):
    """
    Choose a random nonempty proper subset of indices
    Note: subset_range is not guaranteed to hold for edge cases
    """
    assert n >= 2

    min_frac, max_frac = subset_range
    min_size = math.ceil(n * min_frac)
    max_size = math.floor(n * max_frac)

    # Make sure: 1 <= min_size <= max_size <= n - 1
    min_size = min(max(1, min_size), n - 1)
    max_size = min(max(min_size, max_size), n - 1)

    rand_size = random.randint(min_size, max_size)
    subset = random.sample(range(n), rand_size)
    return subset
