"""  Functions for deduplication of sketches at the split level and between splits. """


def get_key(sketch):
    """ Return a unique key. Assume entities and points in entities are sorted. """
    return "".join(sketch["entities"])


def deduplicte_split(sketches):
    deduped_sketches = []
    seen_keys = set()
    for sketch in sketches:
        key = get_key(sketch)
        if key not in seen_keys:
            deduped_sketches.append(sketch)
        seen_keys.add(key)
    return seen_keys, deduped_sketches


def deduplicate_splits(split_to_sketches):
    """ Deduplicate sketches in each split and remove overlap between splits. """
    split_to_count = {split: len(sketches) for split, sketches in split_to_sketches.items()}
    print("Starting deduplication: ", split_to_count)
    # Deduplicate each split
    split_to_keys = {}
    for split, sketches in split_to_sketches.items():
        before_count = len(sketches)
        keys, deduped_sketches = deduplicte_split(sketches)
        split_to_sketches[split] = deduped_sketches
        split_to_keys[split] = keys

        after_count = len(deduped_sketches)
        print(f"Deduplicate {split} - removed {before_count - after_count} sketches ({before_count} to {after_count})")

    # Remove train sketches from val
    before_count = len(split_to_sketches["val"])
    deduped_val_sketches = []
    for sketch in split_to_sketches["val"]:
        key = get_key(sketch)
        if key not in split_to_keys["train"]:
            deduped_val_sketches.append(sketch)
    split_to_sketches["val"] = deduped_val_sketches

    after_count = len(split_to_sketches["val"])
    print(f"Deduplicate val from train - removed {before_count - after_count} sketches ({before_count} to {after_count})")

    # Remove train and val sketches from test
    before_count = len(split_to_sketches["test"])
    deduped_test_sketches = []
    for sketch in split_to_sketches["test"]:
        key = get_key(sketch)
        if key not in split_to_keys["train"] and key not in split_to_keys["val"]:
            deduped_test_sketches.append(sketch)
    split_to_sketches["test"] = deduped_test_sketches

    after_count = len(split_to_sketches["test"])
    print(f"Deduplicate test from val and train - removed {before_count - after_count} sketches ({before_count} to {after_count})")

    split_to_count = {split: len(sketches) for split, sketches in split_to_sketches.items()}
    print("After deduplication: ", split_to_count)

    return split_to_sketches
