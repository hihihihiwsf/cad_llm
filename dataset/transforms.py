import random
import re
import math

def add_quantized(example, n_bits):
    """
    Convert normalized vertices to discrete values in [-(n_bits-1)**2, (n_bits-1)**2 - 1].
    e.g. n_bits=6: [-32, 31]
    This was used in first experiment.
    """
    quantize_range = 2 ** n_bits - 1
    example['quantized_vs'] = (example['vertices'] * quantize_range).astype("int32")
    return example


def add_entities(example):
    """
    Add 'entities' - a list of lists combining the information in vertices and curves
    """
    vertices = example['quantized_vs']
    curves = example['curves']
    entities = [[tuple(vertices[i - 1]) for i in c if i] for c in curves]
    example['entities'] = entities
    return example


def add_random_input_output(example, subset_range):
    entities = example['entities']
    rand_indices = choose_random_input_output_indices(len(entities), subset_range=subset_range)

    example['subset_entities'] = [entities[i] for i in rand_indices['subset']]
    example['completion_entities'] = [entities[i] for i in rand_indices['completion']]
    example['output_entities'] = [entities[i] for i in rand_indices['output']]

    example['input'] = repr_entities(example['subset_entities'])
    example['output'] = repr_entities(example['output_entities'])
    return example


def choose_random_input_output_indices(n, subset_range):
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


def repr_entities(entities):
    entities = sorted(entities)
    return ";".join((",".join((f"{x},{y}" for x, y in ent)) for ent in entities)) + ';'
