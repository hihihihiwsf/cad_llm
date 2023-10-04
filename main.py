"""
Train a CAD LLM model
"""


try:
    import comet_ml  # Import before torch
except ImportError:
    pass
# from dataset.sg_dataset_visrecon import get_sketchgraphs_dataloader
from dataset.sg_dataset_for_constraint import get_sketchgraphs_dataloader, SketchDataModule
#from models.byt5 import ByT5Model
from models import conditional_vl_align, conditional_vision_only, vlt5, vlt5_v2_tri
from models.vl_t5_biencoder import VLT5Model
from models.vis_recon import VisRecon
from torch.utils.data import DataLoader
from util import get_loggers, get_checkpoint_callbacks, get_total_train_steps
from args.main_args import get_training_args
from pathlib import Path
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor
import os
from pytorch_lightning.strategies import DDPStrategy

from transformers import AutoTokenizer

def main():
    """Entry point for our training script"""
    args = get_training_args()
    
    #fabric = Fabric()
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
    tokenizer=AutoTokenizer.from_pretrained(args.model_name)

    print("Loading data...")
    sketchdata = SketchDataModule(tokenizer, args)
    num_train_batches = len(sketchdata.train_dataloader())
    num_gpus = torch.cuda.device_count()
    total_train_steps = get_total_train_steps(num_train_batches, num_gpus, args.epochs)

    print("Loading model...")

    if args.arch == "conditional_vision_only":
        architecture = conditional_vision_only
    elif args.arch == "conditional_vl_align":
        architecture = conditional_vl_align
    
    architecture = vlt5_v2_tri
    if not args.untrained_model:
        model = architecture.ByT5Model(args=args, vit_mae=None, num_train_steps=total_train_steps)
        #model = model.load_from_checkpoint('s3://cad-llm-katzm/jobs/sifan-vit-mae-pd-14-precision16-07-09-23-1627/checkpoints/model/vit_mae_pd_14_precision16/last.ckpt')  #('s3://cad-llm-katzm/jobs/sifan-vlt5-fp16-adafactor-specialtoken-07-11-23-1544/checkpoints/model/vlt5_fp16_adafactor_specialtoken/last.ckpt')
    else:
        print("train_mae", args.untrained_model)
        model = VisRecon(args=args)
        model = model.load_from_checkpoint('s3://cad-llm-katzm/jobs/sifan-mae-ps-32-scratch-dm-07-05-23-1623/checkpoints/model/mae_ps_32_scratch_dm/best.ckpt')
        
    
    # train_dataloader = get_sketchgraphs_dataloader(tokenizer=tokenizer, args=args, split="train", shuffle=True)
    # val_dataloader = get_sketchgraphs_dataloader(tokenizer=tokenizer, args=args, split="val", shuffle=False)
    # test_dataloader = get_sketchgraphs_dataloader(tokenizer=tokenizer, args=args, split="test", shuffle=False)

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
        precision='16',
        check_val_every_n_epoch=args.val_every_n_epoch,
        gradient_clip_val=1.0, 
        gradient_clip_algorithm="value",
        #limit_train_batches=0.01,
        # limit_val_batches=0.1,
    )
    
    if not args.eval: 
        # tuner = Tuner(trainer)
        # lr_finder=tuner.lr_find(model)
        # print(lr_finder.results)
        # fig = lr_finder.plot(suggest=True)
        # fig.savefig('lr_finder.png')
        # new_lr = lr_finder.suggestion()
        # model.hparams.lr = new_lr
        
        print("Start training")
        trainer.fit(model, datamodule=sketchdata) #, ckpt_path='s3://cad-llm-katzm/jobs/sifan-sg-multimodal-v2-triloss-09-06-23-2344/checkpoints/model/sg_multimodal_v2_triloss/best.ckpt')
        trainer.test(model, dataloaders=sketchdata.test_dataloader(), ckpt_path='best')
       
    else:
        # loading the model from exp_name/best.ckpt
        print("Start evaluating")
        ckpt_dir = args.checkpoint_dir + "/{}/checkpoints/best.ckpt".format(args.exp_name)
        #ckpt_path='s3://cad-llm-katzm/jobs/sifan-sg-multimodal-09-05-23-1459/checkpoints/model/sg_multimodal/best.ckpt'
        #ckpt_path = 's3://cad-llm-katzm/jobs/sifan-sg-multimodal-v2-triloss-09-06-23-2344/checkpoints/model/sg_multimodal_v2_triloss/best.ckpt'
        #ckpt_path = '/home/ubuntu/sifan/results/conditional_vl_align/best.ckpt'
        #ckpt_path = 's3://cad-llm-katzm/jobs/sifan-precise-image-conditional-vision-only-09-20-23-1129/checkpoints/model/precise_image_conditional_vision_only/best.ckpt'
        ckpt_path = '/home/ubuntu/sifan/results/test_constraint_vlt5/best.ckpt'
        trainer.test(model, ckpt_path=ckpt_path, dataloaders=sketchdata.test_dataloader())


if __name__ == "__main__":
    main()