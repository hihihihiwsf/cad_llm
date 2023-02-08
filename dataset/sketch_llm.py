from dataset.utils import get_quantized, choose_random_subset
from dataset.entity_llm import EntityLLM
import random


class SketchLLM:
    def __init__(self, sketch_dict, quantize_n_bits):
        self.curves = sketch_dict["curves"]
        self.vertices = sketch_dict["vertices"]
        self.name = sketch_dict["name"]

        if quantize_n_bits:
            self.vertices = get_quantized(self.vertices, quantize_n_bits)

        # Lazy eval
        self.entities = None

        # Random input/output, overridden each call to (epoch)
        self.input_indices = None
        self.output_indices = None
        self.input_string = None
        self.output_string = None

    def add_entities(self):
        entities = []
        for curve in self.curves:
            points = [list(self.vertices[i - 1]) for i in curve if i]
            entities.append(EntityLLM(points=points))
        self.entities = sorted(entities, key=lambda ent: ent.points)

    def add_random_io_indices(self, subset_range):
        """
        Choose random input and output indices. Override values from previous calls.
        """
        n = len(self.entities)
        self.input_indices = choose_random_subset(n, subset_range=subset_range)
        self.input_indices.sort()
        completion_indices = [i for i in range(n) if i not in self.input_indices]
        self.output_indices = random.sample(completion_indices, 1)

    def get_input_output_strings(self):
        """
        Return text representation for input and output entities based on already chosen indices
        """
        if not self.input_indices or not self.output_indices:
            return None
        input_string = "".join(self.entities[i].to_string() for i in self.input_indices)
        output_string = "".join(self.entities[i].to_string() for i in self.output_indices)
        return input_string, output_string

    def get_completion_strings(self):
        completion_indices = [i for i in range(len(self.entities)) if i not in self.input_indices]
        return set(self.entities[i].to_string() for i in completion_indices)

    def generate_random_input_output(self, subset_range):
        if not self.entities:
            self.add_entities()
        self.add_random_io_indices(subset_range=subset_range)
        return self.get_input_output_strings()
