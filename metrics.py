import torch


def calculate_accuracy_samples(labels, samples):
    """
    Count number of exact matches of decoded and sorted entities
    """
    bsz = len(labels)

    count_accurate = 0
    for idx in range(bsz):
        label_entities = labels[idx]
        for j in range(5):
            sample_entities = samples[idx*5]
    #for label_entities, sample_entities in zip(labels, samples):
            label_entities = tuple(sorted([ent for ent in label_entities if ent]))
            sample_entities = tuple(sorted([ent for ent in sample_entities if ent]))
            if label_entities == sample_entities:
                count_accurate += 1
                break
    return count_accurate / len(labels)


def calculate_first_ent_accuracy_samples(labels, samples):
    """
    Calculate accuracy of first predicted entity
    """
    count_accurate = 0
    bsz = len(labels)
    for idx in range(bsz):
        label_entities = labels[idx]
        for j in range(5):
            sample_entities = samples[idx*5+j]
    #for label_entities, sample_entities in zip(labels, samples):
        if not sample_entities:
            continue
        first_entity = sample_entities[0]
        if first_entity and first_entity in label_entities:
            count_accurate += 1
    return count_accurate / len(labels)

def calculate_accuracy(labels, samples):
    """
    Count number of exact matches of decoded and sorted entities
    """
    count_accurate = 0
    for label_entities, sample_entities in zip(labels, samples):
        label_entities = tuple(sorted([ent for ent in label_entities if ent]))
        sample_entities = tuple(sorted([ent for ent in sample_entities if ent]))
        if label_entities == sample_entities:
            count_accurate += 1
    return count_accurate / len(labels)

def calculate_first_ent_accuracy(labels, samples):
    """
    Calculate accuracy of first predicted entity
    """
    count_accurate = 0
    for label_entities, sample_entities in zip(labels, samples):
        if not sample_entities:
            continue
        first_entity = sample_entities[0]
        if first_entity and first_entity in label_entities:
            count_accurate += 1
    return count_accurate / len(labels)


def calculate_validity(batch_sample_curves):
    curve_count = 0
    valid_count = 0

    for sample_curves in batch_sample_curves:
        curve_count += len(sample_curves)
        valid_sample_curves = [curve for curve in sample_curves if curve and curve.good]
        valid_count += len(valid_sample_curves)

    percent_valid = valid_count / max(curve_count, 1)

    return percent_valid
