import comet_ml
import time
from functools import partial
import torch


def count_accurate(labels, samples):
    batch_size, num_tokens = labels.shape
    # Replace -100 masked tokens with 0 to match pad token in samples
    labels[labels == -100] = 0
    # Truncate/Pad samples, skip initial pad token
    samples = samples[:, 1:num_tokens+1]
    pad_amount = num_tokens - samples.shape[1]
    assert pad_amount >= 0, f"Negative pad amount {pad_amount}"
    padded_samples = torch.nn.functional.pad(samples, (0, pad_amount), mode='constant', value=0)
    return int(torch.sum((padded_samples == labels).all(dim=1)))


def compute_metrics(eval_pred, exp_name, model, dataloader):
    start_time = time.time()
    # eval_pred is of type transformers.EvalPrediction
    experiment = comet_ml.get_global_experiment()
    if experiment:
        experiment.set_name(exp_name)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model.eval()
    accurate_count = 0
    count = 0
    for batch in dataloader:
        # Generate samples to test accuracy (do not send labels in for generation)
        labels = batch["labels"].to(device)
        samples = model.generate(input_ids=batch["input_ids"].to(device),
                                 attention_mask=batch["attention_mask"].to(device),
                                 do_sample=False,
                                 max_new_tokens=labels.shape[1])
        accurate_count += count_accurate(labels=labels, samples=samples)
        count += labels.shape[0]
    model.train()

    accuracy = accurate_count / count

    stats = {"accuracy": accuracy}
    print("Eval stats:", stats)
    print(f"Compute metrics time: {int(time.time() - start_time)} seconds")
    return stats


def get_compute_metrics(exp_name, model, dataloader):
    return partial(compute_metrics, exp_name=exp_name, model=model, dataloader=dataloader)
