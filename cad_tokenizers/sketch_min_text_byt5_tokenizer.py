import json

from transformers import ByT5Tokenizer

from preprocess.preprocess_utils import point_entity_from_flat_points

from cad_tokenizers.sketch_tokenizer_base import SketchTokenizerBase


class SketchMinTextTokenizerBase(SketchTokenizerBase):
    """
    Abstract class that use minimal text representation to tokenize sketch entities
    Usage: subclass this class and inherit from a tokenizer class (e.g. ByT5Tokenizer)

    For example this list of two entities:
    [[[0, 1], [63, 1]], [[0, 61], [63, 61]]]

    can be converted to a minimal text representation of the form
    "0,1,63,61;0,61,63,61;"

    For prompts we add a "|" at the end of the string
    """
    NUM_COORDS = 64

    def entities_to_str(self, entities, is_prompt=False):
        res = "".join([self.entity_to_str(entity) for entity in entities])
        if is_prompt:
            res += "|"
        return res

    def entity_to_str(self, entity):
        flat_ent = [coord for point in entity for coord in point]
        assert all(0 <= coord < self.NUM_COORDS for coord in flat_ent)

        return ",".join([str(coord) for coord in flat_ent]) + ";"

    def str_to_entities(self, text, sort=True, safe=True, is_prompt=False):
        try:
            if is_prompt:
                assert text[-1] == "|" and text.count("|") == 1, "Prompt must end with |"
                text = text[:-1]

            entity_texts = text.split(";")
            entities = [self.str_to_entity(text=entity_text, sort=sort) for entity_text in entity_texts if entity_text]
            if sort:
                entities = sorted(entities)

        except ValueError as e:
            if safe:
                print(f"Error parsing {text}: {e}")
                entities = []
            else:
                raise e

        return entities

    def str_to_entity(self, text, sort=True):
        # convert to list of ints
        flat_points = [int(num_str) for num_str in text.split(",") if num_str]

        # assert all coordinates are valid
        assert all(0 <= coord < self.NUM_COORDS for coord in flat_points)

        entity = point_entity_from_flat_points(flat_points, sort=sort)

        return entity


class SketchMinTextByt5Tokenizer(ByT5Tokenizer, SketchMinTextTokenizerBase):
    """
    Usage: SketchMinTextByt5Tokenizer.from_pretrained(model_name)
    """
    pass
