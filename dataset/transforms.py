import random
import re


def to_quantized(example, n_bits):
    """
    Convert normalized vertices to discrete values in [-(n_bits-1)**2, (n_bits-1)**2 - 1].
    e.g. n_bits=6: [-32, 31]
    This was used in first experiment.
    """
    quantize_range = 2 ** n_bits - 1
    example['vertices'] = (example['vertices'] * quantize_range).astype("int32")
    return example


def add_entities(example):
    """
    Add 'entities' - a list of lists combining the information in vertices and curves
    """
    vertices = example['vertices']
    curves = example['curves']
    entities = [[tuple(vertices[i - 1]) for i in c if i] for c in curves]
    example['entities'] = entities
    return example


def add_input_output(example):
    entities = example['entities']

    # Choose nonempty subset and split to input and output
    rand_size = random.randint(1, len(entities))
    subset = random.sample(entities, rand_size)

    input_entities = sorted(subset[:-1])
    output_entities = [subset[-1]]  # List with one element

    example['input'] = repr_entities(input_entities)
    example['output'] = repr_entities(output_entities)

    return example


def repr_entities(entities):
    ent_strings = [re.sub('[\[\]\s]', '', repr(ent)) + ';' for ent in entities]
    return "".join(ent_strings)
