# Decorator function to register tokenizers
def register_model(model_name):
    def decorator(cls):
        model_name_to_cls[model_name] = cls
        return cls

    return decorator


# Dictionary to store registered models
model_name_to_cls = {}


def get_model_cls(tokenizer_name):
    return model_name_to_cls[tokenizer_name]


def get_all_model_names():
    return model_name_to_cls.keys()
