from transforms import repr_entities, choose_random_input_output_indices

string_entities = '-11,29,-6,-23;-6,-23,5,-31,11,-18;'
entities = [
    ((-11, 29), (-6, -23)),
    ((-6, -23), (5, -31), (11, -18)),
]


def test_repr_entities():
    assert string_entities == repr_entities(entities), repr_entities(entities)
    print("success - test_repr_entities")


def test_choose_random_input_output_indices():
    n = 10
    indices = choose_random_input_output_indices(n, (0, 1))
    subset_indices = set(indices['subset'])
    completion_indices = set(indices['completion'])
    output_indices = set(indices['output'])

    assert subset_indices.intersection(completion_indices) == set()
    assert subset_indices.union(completion_indices) == set(range(n))
    assert output_indices.intersection(completion_indices) == output_indices

    choose_random_input_output_indices(n, (0.4, 0.6))
    assert 0.4 <= len(indices['subset']) / n <= 0.6

    print("success - test_choose_random_input_output_indices")


if __name__ == '__main__':
    test_repr_entities()
    test_choose_random_input_output_indices()
