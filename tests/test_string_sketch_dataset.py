import unittest

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset.sketch_strings_dataset import get_sketch_strings_dataset
from dataset.sketch_strings_collator import SketchStringsCollator


class TestStringSketchDataset(unittest.TestCase):
    def test_get_sketch_strings_dataset(self):
        path = "mock_entities_data/"

        tokenizer = AutoTokenizer.from_pretrained("google/byt5-base")
        dataset = get_sketch_strings_dataset(path,  min_split_ratio=0.2, max_split_ratio=0.8)

        self.assertEqual(dataset.keys(), {"test", "train", "val"})

        example = dataset['val'][0]
        expected_keys = ["input_text", "output_text"]
        self.assertTrue(all(key in example.keys() for key in expected_keys))

        batch_size = 2
        collate_fn = SketchStringsCollator(tokenizer=tokenizer, max_length=96)
        dataloader = DataLoader(dataset['val'], batch_size=batch_size, shuffle=False, num_workers=1,
                                collate_fn=collate_fn)

        expected_keys = {"input_text", "output_text", "input_ids", "attention_mask", "labels"}
        batch = next(iter(dataloader))
        self.assertEqual(set(batch.keys()), expected_keys)
        self.assertEqual(len(batch["input_ids"]), batch_size)
