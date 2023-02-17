import random
import math


def get_quantized(vertices, n_bits):
    """
    Convert normalized vertices to discrete values in [-2**(n_bits-1), 2**(n_bits-1) - 1].
    e.g. n_bits=6: [-32, 31]
    This was used in first experiment.
    """
    quantize_range = 2 ** n_bits - 1
    return (vertices * quantize_range).astype("int32")


def get_quantized_range(quantize_n_bits):
    return range(-2 ** (quantize_n_bits - 1), 2 ** (quantize_n_bits - 1))


def choose_random_io_indices(n, subset_range):
    """
    Choose a random nonempty subset of curves for input, and one curve from the remaining for the output.
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
    completion = [i for i in range(n) if i not in subset]
    output = random.sample(completion, 1)
    return dict(subset=subset, completion=completion, output=output)
