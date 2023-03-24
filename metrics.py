import torch


def calculate_accuracy(labels, samples):
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
    accuracy = (padded_samples == labels).all(dim=1).float()
    return torch.mean(accuracy)


def calculate_first_ent_accuracy(string_labels, string_samples):
    """
    Calculate accuracy of first predicted entity
    """
    count_accurate = 0
    for label_string, sample_string in zip(string_labels, string_samples):
        first_entity = sample_string.split(";")[0].replace(" ", "")
        label_entities = label_string.replace(" ", "").split(";")
        label_entities = set(ent for ent in label_entities if ent)
        if first_entity and first_entity in label_entities:
            count_accurate += 1
    return count_accurate / len(string_labels)


def calculate_validity(batch_sample_curves):
    curve_count = 0
    valid_count = 0

    for sample_curves in batch_sample_curves:
        curve_count += len(sample_curves)
        valid_sample_curves = [curve for curve in sample_curves if curve and curve.good]
        valid_count += len(valid_sample_curves)

    percent_valid = valid_count / curve_count

    return percent_valid
