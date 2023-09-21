from dataset.byt5_datamodule import Byt5DataModule
import argparse
from cad_tokenizers.cad_tokenizers_utils import get_tokenizer_cls
import pandas as pd


def main(dataset_path, model_name, tokenizer_name):
    tokenizer_cls = get_tokenizer_cls(tokenizer_name)
    datamodule = Byt5DataModule(
        model_name=model_name,
        batch_size=1,
        max_length=10000,
        min_ratio=0.2,
        max_ratio=0.8,
        input_s3_bucket="",
        dataset_path=dataset_path,
        num_dataloader_workers=8,
        tokenizer_cls=tokenizer_cls
    )

    datamodule.setup(stage="fit")

    val_dataloaders = datamodule.val_dataloader()

    for val_dataloader, name in zip(val_dataloaders, ["random", "20", "40", "60", "80"]):
        lengths = []
        for batch in val_dataloader:
            lengths.append(len(batch["input_ids"][0]))

        # get quantiles
        df = pd.DataFrame(lengths)
        quantiles = df.quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99])

        print(name)
        print(quantiles)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True, type=str, help="Path to save dataset")
    parser.add_argument("--model_name", required=True, type=str, help="Name of huggingface model")
    parser.add_argument("--tokenizer_name", required=True, type=str,
                        help="Name of tokenizer as registered in cad_tokenizers_utils.py")

    args = parser.parse_args()

    main(dataset_path=args.dataset_path, model_name=args.model_name, tokenizer_name=args.tokenizer_name)
