from utils import choose_random_subset
from sketch_llm import SketchLLM
import numpy as np

sketch_dict = {
    'name': 'val_00043061',
    'vertices': np.array([
        [0.5,  0.28],
        [-0.5,  0.28],
        [0.5, -0.28],
        [-0.5, -0.28]
    ]),
    'curves': np.array([
        [1, 2, 0, 0],
        [3, 4, 0, 0],
        [1, 3, 0, 0],
        [2, 4, 0, 0]
    ]),
}

expected_entities = ['-31,-17,-31,17;', '-31,-17,31,-17;', '-31,17,31,17;', '31,-17,31,17;']


def test_sketch():
    sketch = SketchLLM(sketch_dict, quantize_n_bits=6)
    sketch.add_entities()
    all_sorted_entity_strings = [ent.to_string() for ent in sketch.entities]
    assert all_sorted_entity_strings == expected_entities

    input_text, output_text = sketch.generate_random_input_output(subset_range=[0, 1])
    input_entities = set([s + ";" for s in input_text.split(";") if s])
    completion_entities = sketch.get_completion_strings()

    assert input_entities.union(completion_entities) == set(all_sorted_entity_strings)
    assert input_entities.intersection(completion_entities) == set()
    assert output_text in completion_entities

    print("success - test_sketch")


def test_choose_random_subset():
    n = 10
    subset = choose_random_subset(n, (0, 1))
    assert 1 < len(subset) < n, f"len(subset) = {len(subset)}"

    subset = choose_random_subset(n, (0.4, 0.6))
    assert 0.4 <= len(subset) / n <= 0.6

    print("success - test_choose_random_input_output_indices")


if __name__ == '__main__':
    test_choose_random_subset()
    test_sketch()
