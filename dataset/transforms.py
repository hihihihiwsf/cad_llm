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


def add_subset(example, subset_range):
    """
    Choose a random subset of at lease two curves (we assume example has at least two curves)
    Note: subset_range is not guaranteed to hold for edge cases
    """
    entities = example['entities']

    min_frac, max_frac = subset_range
    min_size = math.ceil(len(entities) * min_frac)
    max_size = math.floor(len(entities) * max_frac)

    # Make sure: 2 <= min_size <= max_size <= len(entities)
    min_size = min(max(2, min_size), len(entities))
    max_size = min(max(min_size, max_size), len(entities))

    rand_size = random.randint(min_size, max_size)
    example['subset'] = random.sample(entities, rand_size)


def add_input_output(example):
    subset = example['subset']
    input_entities = sorted(subset[:-1])
    output_entities = [subset[-1]]  # List with one element

    example['input'] = repr_entities(input_entities)
    example['output'] = repr_entities(output_entities)

    return example


def repr_entities(entities):
    return ";".join((",".join((f"{x},{y}" for x, y in ent)) for ent in entities)) + ';'
