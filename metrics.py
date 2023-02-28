import torch


def count_accurate(labels, samples):
    """
    Count number of exact matches without decoding
    """
    batch_size, num_tokens = labels.shape
    # Replace -100 masked tokens with 0 to match pad token in samples
    labels[labels == -100] = 0
    # Truncate/Pad samples, skip initial pad token
    samples = samples[:, 1:num_tokens+1]
    pad_amount = num_tokens - samples.shape[1]
    assert pad_amount >= 0, f"Negative pad amount {pad_amount}"
    padded_samples = torch.nn.functional.pad(samples, (0, pad_amount), mode='constant', value=0)
    return int(torch.sum((padded_samples == labels).all(dim=1)))


def get_accuracy_counts(samples, labels, tokenizer):
    """
    Decode samples and labels and count number of exact matches as well as the number
    of times that at least one suggested entity is correct, i.e. appears in the label
    """
    # Replace -100 masked tokens with 0 before decoding
    labels[labels == -100] = 0

    decoded_samples = tokenizer.batch_decode(samples, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    any_correct_count = 0
    accurate_count = 0
    for sample, label in zip(decoded_samples, decoded_labels):
        sample_entities = set([s.replace(" ", "") + ';' for s in sample.split(';') if s])
        label_entities = set([s.replace(" ", "") + ';' for s in label.split(';') if s])
        intersection = sample_entities.intersection(label_entities)
        if len(intersection) > 0:
            any_correct_count += 1
        if len(intersection) == len(sample_entities) == len(label_entities):
            accurate_count += 1
    return dict(any_correct_count=any_correct_count, accurate_count=accurate_count)
