"""
Train a CAD LLM model
"""


try:
    import comet_ml  # Import before torch
except ImportError:
    pass
from dataset.sg_dataset_visrecon import get_sketchgraphs_dataloader
# from dataset.sg_dataset import get_sketchgraphs_dataloader
# from models.byt5 import ByT5Model
from models.vis_recon import VisRecon
from torch.utils.data import DataLoader
from util import get_loggers, get_checkpoint_callbacks
from args.main_args import get_training_args
from pathlib import Path
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor
import os
from pytorch_lightning.strategies import DDPStrategy

def main():
    """Entry point for our training script"""
    args = get_training_args()
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # torch.set_float32_matmul_precision('high')
    
    results_dir = Path(args.results_dir) / args.exp_name
    if not results_dir.exists():
        results_dir.mkdir(parents=True)

    samples_dir = results_dir / "samples"
    args.samples_dir = str(samples_dir)
    if not samples_dir.exists():
        samples_dir.mkdir()

    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)

    loggers = get_loggers(args=args, log_dir=results_dir)

    # pl.utilities.seed.seed_everything(args.seed)
    pl.seed_everything(args.seed)

    print("Loading model...")

    from transformers import ViTMAEForPreTraining 
    # vitmae_model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
    model = VisRecon(args=args)
    model.tokenizer = None

    print("Loading data...")
    train_dataloader = get_sketchgraphs_dataloader(tokenizer=model.tokenizer, args=args, split="train", shuffle=True)
    val_dataloader = get_sketchgraphs_dataloader(tokenizer=model.tokenizer, args=args, split="val", shuffle=False)

    call_backs = get_checkpoint_callbacks(log_dir=results_dir, all_checkpoint_dir=checkpoint_dir,
                                          using_sagemaker=args.using_sagemaker)

    call_backs.append(LearningRateMonitor(logging_interval='step'))

    print("Training the model...")
    log_every_n_steps = 1000
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"

    trainer = pl.Trainer(
        callbacks=call_backs,
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=DDPStrategy(find_unused_parameters=True),
        logger=loggers,
        max_epochs=args.epochs,
        log_every_n_steps=log_every_n_steps,
        # resume_from_checkpoint=None,
        precision='16',
        check_val_every_n_epoch=args.val_every_n_epoch,
        # limit_train_batches=0.01,
        # limit_val_batches=0.1,
    )
    if not args.eval: 
        #trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        trainer.validate(model, dataloaders=val_dataloader)
    else:
        # loading the model from exp_name/best.ckpt
        #ckpt_dir = args.checkpoint_dir + "/{}/best.ckpt".format(args.exp_name)
        ckpt_dir = '/Tmp/sifan/cad/best.ckpt'
        trainer.validate(model, ckpt_path=ckpt_dir, dataloaders=val_dataloader)


if __name__ == "__main__":
    main()