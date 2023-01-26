import comet_ml
from functools import partial
import torch


def sample_tokens_from_predictions(predictions, temp):
    # Sample from model output
    decoder_output, _ = predictions  # (decoder_output, maybe_encoder_output??)
    batch_size, label_token_count, vocab_size = decoder_output.shape
    # maybe_encoder_output has shape (batch_size, input_token_count, d_model)

    batch_flat_token_scores = torch.from_numpy(decoder_output).view(-1, vocab_size)
    batch_flat_token_scores = batch_flat_token_scores / temp

    probs = torch.nn.functional.softmax(batch_flat_token_scores, dim=-1)
    predicted_tokens = torch.multinomial(probs, num_samples=1).view(batch_size, label_token_count)
    return predicted_tokens


def compute_metrics(eval_pred, exp_name, tokenizer, temp):
    # eval_pred is of type transformers.EvalPrediction
    assert temp > 0, "Not implemented for temp=0"
    experiment = comet_ml.get_global_experiment()
    experiment.set_name(exp_name)
    print("In compute_metrics")
    # Get labels
    labels = eval_pred.label_ids
    predicted_tokens = sample_tokens_from_predictions(eval_pred.predictions, temp=temp)

    token_accuracy = (predicted_tokens.numpy() == labels).mean()

    sample_size = 20
    string_samples = tokenizer.batch_decode(predicted_tokens[:sample_size, :], skip_special_tokens=True)
    print("First 5 batch_decode results", string_samples)
    experiment.log_text(string_samples)

    return {"token_accuracy": token_accuracy}


def get_compute_metrics(exp_name, tokenizer, temp):
    return partial(compute_metrics, exp_name=exp_name, tokenizer=tokenizer, temp=temp)
