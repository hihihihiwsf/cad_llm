from transformers import T5ForConditionalGeneration, AutoTokenizer


def get_byt5_model(checkpoint=None):
    tokenizer = AutoTokenizer.from_pretrained('google/byt5-base')
    checkpoint = checkpoint or 'google/byt5-base'
    model = T5ForConditionalGeneration.from_pretrained(checkpoint)
    model.to('cuda')
    return tokenizer, model
