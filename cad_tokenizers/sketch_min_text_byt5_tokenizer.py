import json

from transformers import ByT5Tokenizer, PreTrainedTokenizer

from preprocess.preprocess_utils import sort_points, point_entity_from_flat_points


class SketchMinTextTokenizerBase(PreTrainedTokenizer):
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

    def batch_decode_to_entities(self, batch, skip_special_tokens=True, sort=True):
        batch_texts = self.batch_decode(batch, skip_special_tokens=skip_special_tokens)
        batch_preds = [self.str_to_entities(text, sort=sort) for text in batch_texts]
        return batch_preds


class SketchMinTextByt5Tokenizer(ByT5Tokenizer, SketchMinTextTokenizerBase):
    """
    Usage: SketchJsonTokenizer.from_pretrained(model_name)
    """
    pass
