from cad_tokenizers.sketch_single_token_byt5_tokenizer import SketchSingleTokenByt5Tokenizer
from cad_tokenizers.sketch_min_text_byt5_tokenizer import SketchMinTextByt5Tokenizer

# TODO: use decorator to register tokenizers
tokenizer_name_to_cls = {
    "single_token_byt5": SketchSingleTokenByt5Tokenizer,
    "min_text_byt5": SketchMinTextByt5Tokenizer,
}


def get_tokenizer_cls(tokenizer_name):
    return tokenizer_name_to_cls[tokenizer_name]


def get_all_tokenizer_names():
    return tokenizer_name_to_cls.keys()
