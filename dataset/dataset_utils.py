import random
import numpy as np


def get_random_mask(n, min_ratio, max_ratio):
    """
    Sample a random size for mask and a random mask of size n
    """
    mask_ratio = random.uniform(min_ratio, max_ratio)
    mask_size = round(mask_ratio * n)
    mask_size = min(max(1, mask_size), n - 1)

    mask = np.zeros(n, dtype=bool)
    mask[:mask_size] = 1
    np.random.shuffle(mask)
    return mask


def split_list(items, min_ratio, max_ratio):
    #length_list = [len(item) for item in items]
    mask = get_random_mask(len(items), min_ratio, max_ratio)
    input_items = [item for i, item in enumerate(items) if mask[i]]
    output_items = [item for i, item in enumerate(items) if not mask[i]]
    return input_items, output_items
