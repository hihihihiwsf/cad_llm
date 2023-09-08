import unittest

import numpy as np

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
        val_dataloaders = datamodule.val_dataloader()

        for val_dataloader in val_dataloaders:
            batch = next(iter(val_dataloader))

            expected_keys = ["input_ids", "attention_mask", "labels", "entities", "input_entities", "output_entities"]
            self.assertTrue(all(key in batch.keys() for key in expected_keys))
            self.assertEqual(len(batch["input_ids"]), batch_size)
            self.assertLessEqual(batch["input_ids"].size()[1], max_length)

            actual_num_tokens = batch["attention_mask"][0].sum()
            expected_num_tokens = batch["input_text"][0].count("<") + 1
            self.assertEqual(actual_num_tokens, expected_num_tokens)

            actual_num_tokens = (batch["labels"][0] != -100).sum()
            expected_num_tokens = batch["output_text"][0].count("<") + 1
            self.assertEqual(actual_num_tokens, expected_num_tokens)

    def test_byt5_new_tokens_data_module_validation_ratios(self):
        batch_size = 16
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
        val_dataloaders = datamodule.val_dataloader()

        expected_ratios = [0.2, 0.4, 0.6, 0.8]
        for val_dataloader, expected_ratio in zip(val_dataloaders[1:], expected_ratios):
            batch = next(iter(val_dataloader))
            ents = batch["entities"]
            in_ents = batch["input_entities"]
            out_ents = batch["output_entities"]
            self.assertTrue(all(len(in_ents[i]) + len(out_ents[i]) == len(ents[i]) for i in range(len(ents))))

            all_entities = batch["entities"]
            input_entities = batch["entities"]
            ratios = [len(input_entities[i]) / len(all_entities[i]) for i in range(len(all_entities))]
            ratio = np.average(ratios)

            # Large delta since sample is small and skewed by 0.5 ratio for sketches with two entities
            delta = 0.1
            self.assertAlmostEqual(ratio, expected_ratio, delta=delta)
