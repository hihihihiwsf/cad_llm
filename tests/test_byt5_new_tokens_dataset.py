import unittest

from dataset.byt5_new_tokens_dataset import Byt5NewTokensDataModule


class TestByt5NewTokensDataModule(unittest.TestCase):
    def test_byt5_new_tokens_data_module(self):
        batch_size = 2
        max_length = 128
        ds_path = "mock_entities_data/"

        datamodule = Byt5NewTokensDataModule(
            model_name="google/byt5-small",
            batch_size=batch_size,
            max_length=max_length,
            dataset_path=ds_path,
            num_dataloader_workers=1,
            min_ratio=0.2,
            max_ratio=0.8,
            input_s3_bucket="",
        )

        datamodule.setup(stage="fit")
        val_dataloader = datamodule.val_dataloader()

        batch = next(iter(val_dataloader))
        print(batch)

        expected_keys = ["input_ids", "attention_mask", "labels"]
        self.assertTrue(all(key in batch.keys() for key in expected_keys))
        self.assertEqual(len(batch["input_ids"]), batch_size)
        self.assertLessEqual(batch["input_ids"].size()[1], max_length)

        actual_num_tokens = batch["attention_mask"][0].sum()
        expected_num_tokens = batch["input_text"][0].count("<") + 1
        self.assertEqual(actual_num_tokens, expected_num_tokens)

        actual_num_tokens = (batch["labels"][0] != -100).sum()
        expected_num_tokens = batch["output_text"][0].count("<") + 1
        self.assertEqual(actual_num_tokens, expected_num_tokens)
