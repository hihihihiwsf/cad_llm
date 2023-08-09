import unittest

from transformers import AutoTokenizer

from dataset.syn_constraints_dataset import SynConstraintsDataModule


class TestSynConstraintsDataModule(unittest.TestCase):
    def test_syn_constraints_datamodule(self):
        ds_path = "syn_constraints_test_data/"
        batch_size = 2
        max_length = 128
        mock_args = unittest.mock.Mock(model_name="google/byt5-small", batch_size=batch_size, max_length=max_length,
                                       dataset=ds_path, num_workers=1)
        datamodule = SynConstraintsDataModule(args=mock_args, ray_args=None)

        datamodule.setup(stage="fit")

        dataloader = datamodule.val_dataloader()

        batch = next(iter(dataloader))

        print(batch["input_text"])

        print(batch["output_text"])

        expected_keys = ["input_ids", "attention_mask", "labels"]
        self.assertTrue(all(key in batch.keys() for key in expected_keys))
        self.assertEqual(len(batch["input_ids"]), batch_size)
        self.assertEqual(batch["input_ids"].shape, (batch_size, max_length))
