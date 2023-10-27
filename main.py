"""
Train a CAD LLM model
"""


try:
    import comet_ml  # Import before torch
except ImportError:
    pass
# from dataset.sg_dataset_visrecon import get_sketchgraphs_dataloader
from dataset import sg_dataset_for_constraint, sg_dataset, sg_dataset_imageconditional #import get_sketchgraphs_dataloader, SketchDataModule
#from models.byt5 import ByT5Model
from models import vlt5, vlt5_v2_tri, byt5,vlt5_for_cons_type, vlt5_v2_tri_2_wo_IDL, vlt5_wo_ITC, vlt5_wo_ITC_IDL
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
    if args.constraint_model:
        special_tokens_dict = {'additional_special_tokens':["e" + str(i) for i in range(0, 31)]}
        tokenizer.add_special_tokens(special_tokens_dict)
    

    print("Loading data...")
    if args.constraint_model:
        dataset = sg_dataset_for_constraint
    else:
        dataset = sg_dataset
        
    sketchdata = dataset.SketchDataModule(tokenizer, args)
    
    '''
    tokenized_length
    import matplotlib.pyplot as plt
    all_input_lengths = []
    all_output_lengths = []
    for _, input_lengths, output_lengths in sketchdata.val_dataloader():
            all_input_lengths.extend(input_lengths)
            all_output_lengths.extend(output_lengths)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(all_input_lengths, bins=30, color='blue', alpha=0.7)
    plt.title('Input Token Length Distribution')
    plt.xlabel('Length')
    plt.ylabel('Number of Samples')

    plt.subplot(1, 2, 2)
    plt.hist(all_output_lengths, bins=30, color='red', alpha=0.7)
    plt.title('Output Token Length Distribution')
    plt.xlabel('Length')
    plt.ylabel('Number of Samples')

    plt.tight_layout()
    plt.savefig("val_input_output_length.png")
    '''
    
    num_train_batches = len(sketchdata.train_dataloader())
    num_gpus = torch.cuda.device_count()
    total_train_steps = get_total_train_steps(num_train_batches, num_gpus, args.epochs)

    print("Loading model...")

    if args.arch == "vlt5_for_cons_type":
        architecture = vlt5_for_cons_type
    elif args.arch == "vlt5_v2_tri":  #### multimodal tri loss model
        architecture = vlt5_v2_tri
    elif args.arch == "vlt5_wo_IDL":
        architecture = vlt5_v2_tri_2_wo_IDL
    elif args.arch == "vlt5_wo_ITC":
        architecture = vlt5_wo_ITC
    elif args.arch == "vlt5_wo_ITC_IDL":
        architecture = vlt5_wo_ITC_IDL
    
    if not args.untrained_model:
        model = architecture.ByT5Model(args=args, vit_mae=None, tokenizer=tokenizer, num_train_steps=total_train_steps)
        #model = model.load_from_checkpoint('s3://cad-llm-katzm/jobs/sifan-vit-mae-pd-14-precision16-07-09-23-1627/checkpoints/model/vit_mae_pd_14_precision16/last.ckpt')  #('s3://cad-llm-katzm/jobs/sifan-vlt5-fp16-adafactor-specialtoken-07-11-23-1544/checkpoints/model/vlt5_fp16_adafactor_specialtoken/last.ckpt')
    else:
        print("train_mae", args.untrained_model)
        model = VisRecon(args=args)
        model = model.load_from_checkpoint('s3://cad-llm-katzm/jobs/sifan-mae-ps-32-scratch-dm-07-05-23-1623/checkpoints/model/mae_ps_32_scratch_dm/best.ckpt')
        

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
        
        print("Start training")
        trainer.fit(model, datamodule=sketchdata) #, ckpt_path='/home/ubuntu/sifan/results/contraint_with_embedding/last.ckpt')
        trainer.test(model, dataloaders=sketchdata.test_dataloader())
       
    else:
        # loading the model from exp_name/best.ckpt
        print("Start evaluating")
        ckpt_dir = args.checkpoint_dir + "/{}/checkpoints/best.ckpt".format(args.exp_name)

        #ckpt_path = '/home/ubuntu/sifan/results/vlt5_2_constraint_with_embedding/best.ckpt'
        ckpt_path = 's3://cad-llm-katzm/jobs/sifan-sg-multimodal-v2-triloss-09-06-23-2344/checkpoints/model/sg_multimodal_v2_triloss/best.ckpt'
        trainer.test(model, ckpt_path=ckpt_path, dataloaders=sketchdata.test_dataloader())
        
    all_input_lengths = model.prediction_len
    all_output_lengths = model.target_len
    import matplotlib.pyplot as plt
    # Plot histograms
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(all_input_lengths, bins=30, color='blue', alpha=0.7)
    plt.title('CadLIP Prediction Token Length Distribution')
    plt.xlabel('Length')
    plt.ylabel('Number of Samples')

    plt.subplot(1, 2, 2)
    plt.hist(all_output_lengths, bins=30, color='red', alpha=0.7)
    plt.title('Cadlip Label Token Length Distribution')
    plt.xlabel('Length')
    plt.ylabel('Number of Samples')

    plt.tight_layout()
    print("plt draw images")
    plt.savefig("cadlip_pred_label_distribution.png")
    import pdb;pdb.set_trace()

if __name__ == "__main__":
    main()