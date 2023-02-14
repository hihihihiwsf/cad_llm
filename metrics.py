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
