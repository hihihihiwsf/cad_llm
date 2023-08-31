"""
Train a CAD LLM model
"""


try:
    import comet_ml  # Import before torch
except ImportError:
    pass
# from dataset.sg_dataset_visrecon import get_sketchgraphs_dataloader
from dataset.sg_dataset import get_sketchgraphs_dataloader
from models.byt5 import ByT5Model


from models.vis_recon import VisRecon


from torch.utils.data import DataLoader
from util import get_loggers, get_checkpoint_callbacks, EmbeddingCallback, StringCallback
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
    torch.set_float32_matmul_precision('medium')

    print("Loading model...")

    if not args.untrained_model:
        #model = ByT5Model(args=args, vit_mae=None)
        model = ByT5Model(args=args, vit_mae=None)
        
        # model = model.load_from_checkpoint('s3://cad-llm-katzm/jobs/sifan-vit-mae-pd-14-precision16-07-09-23-1627/checkpoints/model/vit_mae_pd_14_precision16/last.ckpt') # ('s3://cad-llm-katzm/jobs/sifan-vlt5-fp16-adafactor-specialtoken-07-11-23-1544/checkpoints/model/vlt5_fp16_adafactor_specialtoken/last.ckpt')   
    else:
        print("train_mae...", args.untrained_model)
        model = VisRecon(args=args)
        model = model.load_from_checkpoint('s3://cad-llm-katzm/jobs/sifan-mae-ps-32-scratch-dm-07-05-23-1623/checkpoints/model/mae_ps_32_scratch_dm/best.ckpt')
        
    tokenizer=AutoTokenizer.from_pretrained(args.model_name)

    print("Loading data...")
    train_dataloader = get_sketchgraphs_dataloader(tokenizer=tokenizer, args=args, split="train", shuffle=True)
    val_dataloader = get_sketchgraphs_dataloader(tokenizer=tokenizer, args=args, split="val", shuffle=False)
    test_dataloader = get_sketchgraphs_dataloader(tokenizer=tokenizer, args=args, split="test", shuffle=False)

    call_backs = get_checkpoint_callbacks(log_dir=results_dir, all_checkpoint_dir=checkpoint_dir,
                                          using_sagemaker=args.using_sagemaker)

    call_backs.append(LearningRateMonitor(logging_interval='step'))
    
    # embedding_callback = StringCallback()

    print("Training the model...")
    log_every_n_steps = 1000
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"

    trainer = pl.Trainer(
        callbacks=call_backs, #[embedding_callback],
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=DDPStrategy(find_unused_parameters=True),
        logger=loggers,
        max_epochs=args.epochs,
        log_every_n_steps=log_every_n_steps,
        precision='16-mixed',
        check_val_every_n_epoch=args.val_every_n_epoch,
        #resume_from_checkpoint='s3://cad-llm-katzm/jobs/sifan-vlt5-07-07-23-1038/checkpoints/model/vlt5/best.ckpt',  #'s3://cad-llm-katzm/jobs/sifan-mae-ps-32-scratch-07-04-23-2320/checkpoints/best.ckpt',
        # limit_train_batches=0.01,
        # limit_val_batches=0.1,
    )
    if not args.eval: 
        print("Start training")
        ckpt_dir =  '/home/ubuntu/sifan/results/vit_mae_pd_14_precision16/best.ckpt'
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)#, #ckpt_path=ckpt_dir)  #ckpt_path='s3://cad-llm-katzm/jobs/sifan-vit-mae-pd-14-precision16-07-09-23-1627/checkpoints/model/vit_mae_pd_14_precision16/last.ckpt')
        trainer.test(dataloaders=test_dataloader)
    else:
        # loading the model from exp_name/best.ckpt
        print("Start evaluating")
        #ckpt_dir = args.checkpoint_dir + "/{}/checkpoints/best.ckpt".format(args.exp_name)
        ckpt_dir =  's3://cad-llm-katzm/jobs/sifan-max-96-sg-string-vitmae-08-28-23-2227/checkpoints/model/max_96_sg_string_vitmae/best.ckpt' #'/home/ubuntu/sifan/results/vit_mae_pd_14_precision16/best.ckpt'   ##'s3://cad-llm-katzm/jobs/sifan-vl-biloss-07-17-23-1618/checkpoints/model/vl_biloss/best.ckpt' #s3://cad-llm-katzm/jobs/sifan-vlbiloss-05-lmloss-07-19-23-1617/checkpoints/model/vlbiloss_05_lmloss/best.ckpt' #s3://cad-llm-katzm/jobs/sifan-vit-mae-pd-14-precision16-07-09-23-1627/checkpoints/model/vit_mae_pd_14_precision16/best.ckpt'
        trainer.validate(model, ckpt_path=ckpt_dir, dataloaders=test_dataloader)
        
    # print("running end.........")
    # val_pred_string = embedding_callback.val_pred_string
    # val_label_string = embedding_callback.val_label_string
    # train_pred_string = embedding_callback.train_pred_string
    # train_label_string = embedding_callback.train_label_string
    
    # import json
    # from IPython import embed
    # try:
    #     val_pred_dir = args.results_dir +'/'+"val_pred_string.json"
    #     with open(val_pred_dir, 'w') as f:
    #         json.dump(val_pred_string, f)
            
    #     val_label_dir = args.results_dir +'/'+"val_label_string.json"
    #     with open(val_label_dir, 'w') as f:
    #         json.dump(val_label_string, f)
            
    #     train_pred_dir = args.results_dir +'/'+"train_pred_string.json"
    #     with open(val_pred_dir, 'w') as f:
    #         json.dump(train_pred_string, f)
            
    #     train_label_dir = args.results_dir +'/'+"train_label_string.json"
    #     with open(val_label_dir, 'w') as f:
    #         json.dump(train_label_string, f)
    # except:
    #     embed()
        

if __name__ == "__main__":
    main()