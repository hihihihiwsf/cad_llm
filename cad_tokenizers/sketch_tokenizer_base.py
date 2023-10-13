from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


class SketchTokenizerBase(PreTrainedTokenizer):
    """
    Abstract class that use minimal text representation to tokenize sketch entities
    """

    def entities_to_str(self, entities, is_prompt):
        raise NotImplementedError

    def str_to_entities(self, text, sort, safe, is_prompt):
        raise NotImplementedError

    def batch_decode_to_entities(self, batch, skip_special_tokens=True, sort=True):
        batch_texts = self.batch_decode(batch, skip_special_tokens=skip_special_tokens)
        batch_preds = [self.str_to_entities(text, sort=sort) for text in batch_texts]
        return batch_preds
