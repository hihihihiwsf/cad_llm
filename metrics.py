import torch
import numpy as np

from IPython import embed
def calculate_f1(labels, samples):
    """
    Count number of exact matches of decoded and sorted entities
    """
    f1 = []
    eps = 1e-6
    if all(isinstance(sublist, list) and not sublist for sublist in samples):
        return 0,0,0
    for label_entities, sample_entities in zip(labels, samples):
        if not sample_entities:
            continue
        TP = 0
        for ent in sample_entities:
            if ent in label_entities:
                TP += 1
        FP = len(sample_entities) - TP
        FN = len(label_entities) - TP

        precision = (TP / (TP + FP+ eps ))
        recall = (TP / (TP + FN+ eps )) 

        try:
            f1.append(2 * precision * recall / (precision + recall))
        except:
           f1.append(0)
    return np.mean(precision), np.mean(recall), np.mean(f1)


def calculate_first_ent_accuracy(labels, samples):
    """
    Calculate accuracy of first predicted entity
    """
    count_accurate = 0
    for label_entities, sample_entities in zip(labels, samples):
        if not sample_entities:
            continue
        for ent in sample_entities:
            if ent in label_entities:
                count_accurate += 1
                break
        # first_entity = sample_entities[0]
        # if first_entity and first_entity in label_entities:
        #     count_accurate += 1
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


# def calculate_first_ent_accuracy(labels, samples):
#     """
#     Calculate accuracy of first predicted entity
#     """
#     count_accurate = 0
#     for label_entities, sample_entities in zip(labels, samples):
#         if not sample_entities:
#             continue
#         first_entity = sample_entities[0]
#         if first_entity and first_entity in label_entities:
#             count_accurate += 1
#     return count_accurate / len(labels)


def calculate_validity(batch_sample_curves):
    curve_count = 0
    valid_count = 0

    for sample_curves in batch_sample_curves:
        curve_count += len(sample_curves)
        valid_sample_curves = [curve for curve in sample_curves if curve and curve.good]
        valid_count += len(valid_sample_curves)

    percent_valid = valid_count / max(curve_count, 1)

    return percent_valid

def remove_trailing_zeros_from_sublist(sublist):
    """
    Remove trailing zeros from a sublist.
    
    Args:
    sublist (list): A sublist from which to remove trailing zeros.
    
    Returns:
    list: The sublist with trailing zeros removed.
    """
    # Reverse the sublist to find the first non-zero element from the end
    for i in range(len(sublist) - 1, -1, -1):
        if sublist[i] != 0:
            return sublist[:i + 1]
    return [] 

def calculate_vitruvion(loss, batch):
    output_ids = batch['samples'].cpu()
    np_loss = np.array(loss)
    bits_per_primitive = []
    for batch_index, ids in enumerate(output_ids):
        # Find indices where the ID is '31', signifying the end of a primitive
        separator_indices = np.where(ids == 31)[0]
        # Add the starting index and the length of the loss array for this batch
        indices = np.concatenate(([0], separator_indices + 1, [len(loss[batch_index])]))
        
        # Compute the sum of losses for each primitive by slicing the loss array
        sum_loss_per_primitive = [np.sum(np_loss[batch_index][start:end]) for start, end in zip(indices[:-1], indices[1:])]
        sum_loss_per_primitive = remove_trailing_zeros_from_sublist(sum_loss_per_primitive)
        bits_per_primitive.append(sum_loss_per_primitive)
    
    bits_per_sketch = [sum(sublist) for sublist in bits_per_primitive]
    bits_per_primitive = [np.array(a).mean() for a in bits_per_primitive]
    
    return np.array(bits_per_primitive).mean(), np.array(bits_per_sketch).mean()

