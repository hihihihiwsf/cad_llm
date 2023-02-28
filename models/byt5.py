from transformers import T5Config, T5ForConditionalGeneration, AutoTokenizer


def get_byt5_model(checkpoint=None):
    tokenizer = AutoTokenizer.from_pretrained('google/byt5-base')
    checkpoint = checkpoint or 'google/byt5-base'
    model = T5ForConditionalGeneration.from_pretrained(checkpoint)
    return tokenizer, model


def get_new_byt5_model():
    tokenizer = AutoTokenizer.from_pretrained('google/byt5-base')
    config = T5Config.from_pretrained('google/byt5-base')
    model = T5ForConditionalGeneration(config)
    model._init_weights(model)  # maybe redundant
    return tokenizer, model


def get_byt5_small_model(checkpoint=None):
    tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')
    checkpoint = checkpoint or 'google/byt5-small'
    model = T5ForConditionalGeneration.from_pretrained(checkpoint)
    return tokenizer, model
