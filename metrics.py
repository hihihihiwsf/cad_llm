import comet_ml
import time
from functools import partial
import torch


def sample_tokens_from_predictions(predictions, temperature):
    if temperature == 0:
        return predictions.argmax(-1)

    # Sample from model output
    decoder_output, _ = predictions  # (decoder_output, maybe_encoder_output??)
    batch_size, label_token_count, vocab_size = decoder_output.shape
    # maybe_encoder_output has shape (batch_size, input_token_count, d_model)

    batch_flat_token_scores = torch.from_numpy(decoder_output).view(-1, vocab_size)
    batch_flat_token_scores = batch_flat_token_scores / temperature

    probs = torch.nn.functional.softmax(batch_flat_token_scores, dim=-1)
    predicted_tokens = torch.multinomial(probs, num_samples=1).view(batch_size, label_token_count)
    return predicted_tokens


def compute_metrics(eval_pred, exp_name, dataset, tokenizer, temperature):
    start_time = time.time()
    # eval_pred is of type transformers.EvalPrediction
    experiment = comet_ml.get_global_experiment()
    if experiment:
        experiment.set_name(exp_name)

    # Get labels
    labels = eval_pred.label_ids
    predicted_tokens = sample_tokens_from_predictions(eval_pred.predictions, temperature=temperature)

    token_accuracy = (predicted_tokens.numpy() == labels).mean()

    total_correct = 0
    log_sample_size = 4
    for i in range(len(dataset)):
        sample = tokenizer.decode(predicted_tokens[i, :], skip_special_tokens=True)
        completion_strings = dataset.get_completions(index=i)

        correct = sample in completion_strings
        total_correct += int(correct)

        if i < log_sample_size:
            info = f"sample: {sample}\ncompletions: {completion_strings}\ncorrect: {correct}\n"
            print(info)
            if experiment:
                experiment.log_text(info)

    accuracy = total_correct / len(dataset)
    stats = {"accuracy": accuracy, "token_accuracy": token_accuracy}
    print("Eval stats:", stats)
    print(f"Compute metrics time: {int(time.time() - start_time)} seconds")
    return stats


def get_compute_metrics(exp_name, dataset, tokenizer, temperature):
    return partial(compute_metrics, exp_name=exp_name, dataset=dataset, tokenizer=tokenizer, temperature=temperature)
