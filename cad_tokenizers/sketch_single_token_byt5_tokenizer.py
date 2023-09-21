from transformers import ByT5Tokenizer

from preprocess.preprocess_utils import sort_points, point_entity_from_flat_points


class SketchSingleTokenByt5Tokenizer(ByT5Tokenizer):
    """
    ByT5Tokenizer that tokenizes sketch entities of the form:

    [[[0, 1], [63, 1]], [[0, 61], [63, 61]]]

    This tokenizer uses a single token for each coordinate. Entities are separated by <EOE> tokens.
    End of prompt is marked by <EOP> token.
    """

    END_OF_ENT = "<EOE>"
    END_OF_PROMPT = "<EOP>"

    NUM_COORDS = 64
    NEW_TOKENS = [f"<{i}>" for i in range(NUM_COORDS)] + [END_OF_ENT, END_OF_PROMPT]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_tokens(self.NEW_TOKENS)

    def entities_to_str(self, entities, is_prompt=False):
        entity_string_list = [self.entity_to_str(entity) for entity in entities]

        res = self.END_OF_ENT.join(entity_string_list)

        if is_prompt:
            res += self.END_OF_PROMPT

        return res

    def entity_to_str(self, entity):
        assert all(0 <= coord < self.NUM_COORDS for point in entity for coord in point)
        return "".join([f"<{x}><{y}>" for x, y in entity])

    def str_to_entities(self, text, sort=True, safe=True):
        entity_strings = [s.replace(" ", "") for s in text.split(self.END_OF_ENT) if s]

        point_entities = []
        for entity_string in entity_strings:
            point_entity = self.str_to_entity(text=entity_string, sort=sort, safe=safe)

            point_entities.append(point_entity)

        return point_entities

    def str_to_entity(self, text, sort=True, safe=True):
        try:
            flat_points = [int(num_str.split(">")[0]) for num_str in text.split("<") if num_str]

            point_entity = point_entity_from_flat_points(flat_points, sort=sort)
            if sort:
                point_entity = sort_points(point_entity)
        except ValueError as e:
            if safe:
                print(f"Error parsing {text}: {e}")
                return []
            else:
                raise e

        return point_entity

    def batch_decode_to_entities(self, batch, skip_special_tokens=True, sort=True):
        batch_texts = self.batch_decode(batch, skip_special_tokens=skip_special_tokens)
        batch_preds = [self.str_to_entities(text, sort=sort) for text in batch_texts]
        return batch_preds
