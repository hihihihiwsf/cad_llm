import unittest
import json
from cad_tokenizers.cad_tokenizers_utils import tokenizer_name_to_cls


class TestSketchSingleTokenByt5Tokenizer(unittest.TestCase):
    def _test_encode_decode(self, tokenizer_cls):
        data_path = "../mock_entities_data/val.json"

        # load mock data
        with open(data_path, "r") as f:
            data = json.load(f)

        # load tokenizer
        tokenizer = tokenizer_cls.from_pretrained("google/byt5-small")

        for sketch in data:
            tuple_entities = [tuple(tuple(point) for point in ent) for ent in sketch["entities"]]

            encoded = tokenizer.entities_to_str(sketch["entities"], is_prompt=False)
            encoded_and_decoded = tokenizer.str_to_entities(encoded, sort=True, safe=False)

            self.assertEqual(encoded_and_decoded, tuple_entities)

    def test_all_byt5_tokenizers(self):
        for tokenizer_name, tokenizer_cls in tokenizer_name_to_cls.items():
            print(f"Testing tokenizer {tokenizer_name}")
            self._test_encode_decode(tokenizer_cls)
