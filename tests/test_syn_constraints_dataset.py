import unittest

from dataset.syn_constraints_dataset import SynConstraintsDataModule, SynConstraintsPPDataModule


class TestSynConstraintsDataModule(unittest.TestCase):

    def _test_syn_constraints_datamodule(self, data_module_class):
        batch_size = 2
        max_length = 128
        ds_path = "syn_constraints_test_data/"
        kwargs = {
            "model_name": "google/byt5-small",
            "batch_size": batch_size,
            "max_length": max_length,
            "dataset_path": ds_path,
            "num_workers": 1,
        }

        datamodule = data_module_class(**kwargs)
        datamodule.setup(stage="fit")
        dataloader = datamodule.val_dataloader()

        batch = next(iter(dataloader))
        print(batch)

        expected_keys = ["input_ids", "attention_mask", "labels"]
        self.assertTrue(all(key in batch.keys() for key in expected_keys))
        self.assertEqual(len(batch["input_ids"]), batch_size)
        self.assertEqual(batch["input_ids"].shape, (batch_size, max_length))

    def test_syn_constraints_datamodule(self):
        self._test_syn_constraints_datamodule(SynConstraintsDataModule)

    def test_syn_constraints_pp_datamodule(self):
        self._test_syn_constraints_datamodule(SynConstraintsPPDataModule)
