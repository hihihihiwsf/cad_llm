"""
Train a CAD LLM model
"""


try:
    import comet_ml  # Import before torch
except ImportError:
    pass
from dataset.sg_dataset import get_sketchgraphs_dataloader
from models.byt5 import ByT5Model
from models.codet5 import CodeT5Model
from models.multimodel import VLModel

from util import get_loggers, get_checkpoint_callbacks
from args.main_args import get_training_args
from pathlib import Path
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor


def main():
    """Entry point for our training script"""
    args = get_training_args()

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

    ## fsdp accelarator:
    #accelerator = Accelerator()

    print("Loading model...")
    model = CodeT5Model(args=args)
    #model = accelerator.prepare(model)

    print("Loading data...")
    train_dataloader = get_sketchgraphs_dataloader(tokenizer=model.tokenizer, args=args, split="train", shuffle=True)
    val_dataloader = get_sketchgraphs_dataloader(tokenizer=model.tokenizer, args=args, split="val", shuffle=False)

    call_backs = get_checkpoint_callbacks(log_dir=results_dir, all_checkpoint_dir=checkpoint_dir,
                                          using_sagemaker=args.using_sagemaker)

    call_backs.append(LearningRateMonitor(logging_interval='step'))
    
    # from peft.utils.other import fsdp_auto_wrap_policy
    # if getattr(accelerator.state, "fsdp_plugin", None) is not None:
    #     accelerator.state.fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(model)
    
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # lr_scheduler = {
    #             "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, verbose=True), 
    #             "interval": "epoch",
    #             "frequency": 1,
    #         }

    # optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
    #     optimizer, train_dataloader, val_dataloader, lr_scheduler
    # )
 
    
    WORLD_SIZE = torch.cuda.device_count()
    print("Training the model...")
    log_every_n_steps = 1000
    trainer = pl.Trainer(
        callbacks=call_backs,
        accelerator="gpu",  #accelerator,
        devices=WORLD_SIZE,
        precision=16,
        strategy= "fsdp_native", #args.strategy,
        logger=loggers,
        max_epochs=args.epochs,
        log_every_n_steps=log_every_n_steps,
        # resume_from_checkpoint=None,
        check_val_every_n_epoch=args.val_every_n_epoch,
        # limit_train_batches=0.001,
        # limit_val_batches=0.01,
    )
    if not args.eval: 
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    else:
        # loading the model from exp_name/best.ckpt
        ckpt_dir = args.checkpoint_dir + "/{}/best.ckpt".format(args.exp_name)
        trainer.validate(model, ckpt_path=ckpt_dir, dataloaders=val_dataloader)


if __name__ == "__main__":
    main()
