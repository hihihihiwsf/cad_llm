# Decorator function to register tokenizers
def register_tokenizer(tokenizer_name):
    def decorator(cls):
        tokenizer_name_to_cls[tokenizer_name] = cls
        return cls

    return decorator


# Dictionary to store registered tokenizers
tokenizer_name_to_cls = {}


def get_tokenizer_cls(tokenizer_name):
    return tokenizer_name_to_cls[tokenizer_name]


def get_all_tokenizer_names():
    return tokenizer_name_to_cls.keys()
