"""
Train a CAD LLM model
"""


try:
    import comet_ml  # Import before torch
except ImportError:
    pass
# from dataset.sg_dataset_visrecon import get_sketchgraphs_dataloader
from dataset import sg_dataset_for_constraint, sg_dataset, sg_dataset_imageconditional, sg_dataset_visrecon #import get_sketchgraphs_dataloader, SketchDataModule
#from models.byt5 import ByT5Model
from models import vlt5, vlt5_v2_tri, byt5,vlt5_for_cons_type, vlt5_v2_tri_2_wo_IDL, vlt5_wo_ITC, vlt5_wo_ITC_IDL
from models import conditional_vision_only1
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
    
    dataset = sg_dataset_imageconditional
    sketchdata = dataset.SketchDataModule(tokenizer, args)
    
    '''
    #draw some test samples to compare with vitruvion
    model_batch={}
    import json
    import ast
    import numpy as np
    from IPython import embed
    import matplotlib.pyplot as plt
    from compare_vitru_out import convert_circle, save_entities
    from preprocess.preprocessing import center_vertices, center_and_scale, normalize_and_quantize_vertices
    from geometry.visualization import visualize_batch, visualize_sample, visualize_sample_cv, visualize_sample_handraw2
    with open('/u/wusifan/cadllm/vitruvion/virtruvion_test_prefix.json', 'r') as f:
        test_prefix = json.load(f)
    with open('/u/wusifan/cadllm/vitruvion/virtruvion_test_label.json', 'r') as f:
        test_label = json.load(f)
    with open('/u/wusifan/cadllm/vitruvion/virtruvion_test_output.json', 'r') as f:
        test_output = json.load(f)
    
    def center_and_normalize(part_data, tuple_data):
        if not tuple_data:
            return tuple_data
        vertices = np.array([item for sublist in tuple_data for item in sublist])

        normalized_quantized_vertices = normalize_and_quantize_vertices(vertices)
        normalized_quantized_part_data = []
        output_tuples = []
        start = 0
        for t in tuple_data:
            end = start + len(t)
            output_tuples.append(tuple(map(tuple, normalized_quantized_vertices[start:end])))
            if t in part_data:
                normalized_quantized_part_data.append(tuple(map(tuple, normalized_quantized_vertices[start:end])))
            start = end
        return normalized_quantized_part_data,output_tuples


    point_input_strings = [ast.literal_eval(pre) for pre in test_prefix]
    point_label = [ast.literal_eval(pre) for pre in test_label]
    point_output = [ast.literal_eval(pre) for pre in test_output]
        

    point_inputs = [convert_circle(sketch) for sketch in point_input_strings]
    point_labels =[convert_circle(sketch) for sketch in point_label]
    point_outputs =[convert_circle(sketch) for sketch in point_output]
    
    point_inputs = point_inputs[1200:]
    point_labels = point_labels[1200:]
    point_outputs = point_labels[1200:]
    n_point_inputs = [center_and_normalize(tuple_input, tuple_label)[0] for tuple_input, tuple_label in zip(point_inputs, point_labels)]
    n_point_labels = [center_and_normalize(tuple_input, tuple_label)[1] for tuple_input, tuple_label in zip(point_inputs, point_labels)]
    n_point_outputs = [center_and_normalize(tuple_input, tuple_label)[1] for tuple_input, tuple_label in zip(point_outputs, point_labels)]
    
    start = 200

    
    # hand_draw = visualize_sample_handraw2(n_point_labels[150:200], box_lim=64+3)
    # shift_fraction = 6 / 128
    # scale = 0.2
    # shear = 8
    # rotation = 8
    # import torchvision
    # img_affine = torchvision.transforms.RandomAffine(
    #     rotation,
    #     translate=(shift_fraction, shift_fraction),
    #     scale=(1 - scale, 1 + scale),
    #     shear=shear,
    #     fill=255)
    # noisy_hand_imgs = [img_affine(imgs) for imgs in hand_draw]
    
    # prefix_img = visualize_sample_cv(n_point_inputs[start:start+500], box_lim=64+3)
    # label_img = visualize_sample_cv(n_point_labels[start:start+500], box_lim=64+3)
    # for i in range(500):
    #     label_img[i].save(f'label_outputs/{i+start}_label.png')
    #     prefix_img[i].save(f'prefix_outputs/{i+start}_prefix.png')
    #    #noisy_hand_imgs[i].save(f'noisy_outputs/{i}_noisy.png')


    # saved_idx = [3,6,13,17,28,29, 51, 58, 63, 66, 67, 69,80,82, 86, 91, 92,93,101,110,113,126,127,129,130,132]
    # 2th_saved_idx = [18, 23, 26,32, 37, 49, 52, 60, 75, 97, 118, 124, 126, 136, 138, 141,146,151,157,158,164,160,163,165,173,181,213,216,217,220,224]
    th2_saved_idx = [18, 23, 26,32, 37, 49, 52, 60, 75, 97, 118, 124, 126, 136, 138, 141,146,151,157,158,164,160,163,165,173,181,213,216,217,220,224,236,244,247,252,256,257,264,271,275,276,277,279,280,301,302,303,306,309,315,320,322,323,332,336,340,344,347,356,374,376,377,378,379,398,401,404,407,410,416,417,418,419
    ,425,426,431,436,440,443,455,459,463,464,465,466,471,488,493,492,495,496,503,509,510,514,522,533,535,534
    ,538,540,545,548,561,565,575,584,593,596,601,602,604,605,608,611,615,618,621,624,646,650,651,652,663,667
    ,677,681,682,683,697,698]
    saved_inputs=[]
    saved_label=[]
    saved_output = []
    input_strings = []
    label_strings = []
    for i in th2_saved_idx:
        saved_inputs.append(n_point_inputs[i])
        saved_label.append(n_point_labels[i])
        saved_output.append(n_point_outputs[i])
        input_strings.append(';'.join(','.join(str(item) for pair in tup_set for item in pair) for tup_set in n_point_inputs[i]) + ';')
        label_strings.append(';'.join(','.join(str(item) for pair in tup_set for item in pair) for tup_set in n_point_labels[i]) + ';')
    
    prefix_img = visualize_sample_cv(saved_inputs, box_lim=64+3)
    label_img = visualize_sample_cv(saved_label, box_lim=64+3)
    output_img = visualize_sample_cv(saved_output, box_lim=64+3)
    for i in range(56,len(th2_saved_idx)):
        print(i)
        plt.imshow(label_img[i])
        plt.axis('off') 
        plt.savefig(f'pdfs/{i}_label.pdf', bbox_inches='tight')
        plt.imshow(prefix_img[i])
        plt.axis('off') 
        plt.savefig(f'pdfs/{i}_prefix.pdf', bbox_inches='tight')
        plt.imshow(output_img[i])
        plt.axis('off') 
        plt.savefig(f'pdfs/{i}_vitru.pdf', bbox_inches='tight')
        # label_img[i].save(f'pdfs/{i}_label.pdf', 'PDF', resolution=100.0)
        # prefix_img[i].save(f'pdfs/{i}_prefix.pdf', 'PDF', resolution=100.0)
    embed() 
    with open('pdfs/input_strings.json', 'w') as f:
        json.dump(input_strings, f)
    with open('pdfs/label_strings.json', 'w') as f:
        json.dump(label_strings, f)
      
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
    
    architecture=conditional_vision_only1
    if not args.untrained_model:
        model = architecture.ByT5Model(args=args, vit_mae=None, tokenizer=tokenizer, num_train_steps=total_train_steps)
        #model = model.load_from_checkpoint('s3://cad-llm-katzm/jobs/sifan-vit-mae-pd-14-precision16-07-09-23-1627/checkpoints/model/vit_mae_pd_14_precision16/last.ckpt')  #('s3://cad-llm-katzm/jobs/sifan-vlt5-fp16-adafactor-specialtoken-07-11-23-1544/checkpoints/model/vlt5_fp16_adafactor_specialtoken/last.ckpt')
    else:
        print("train_mae", args.untrained_model)
        model = VisRecon(args=args)
        
        #model = model.load_from_checkpoint('s3://cad-llm-katzm/jobs/sifan-mae-ps-32-scratch-dm-07-05-23-1623/checkpoints/model/mae_ps_32_scratch_dm/best.ckpt')
        

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
        limit_val_batches=0.003,
        #limit_test_batches=0.1,
    )
    
    if not args.eval: 
        
        print("Start training")
        trainer.fit(model, datamodule=sketchdata) #, ckpt_path='/home/ubuntu/sifan/results/contraint_with_embedding/last.ckpt')
        #trainer.validate(model, dataloaders=sketchdata.test_dataloader())
       
    else:
        # loading the model from exp_name/best.ckpt
        print("Start evaluating")
        ckpt_dir = args.checkpoint_dir + "/{}/checkpoints/best.ckpt".format(args.exp_name)

        #ckpt_path = '/home/ubuntu/sifan/results/vlt5_2_constraint_with_embedding/best.ckpt'
        #ckpt_path = '/Tmp/sifan/cad/sg_multimodal_v2_triloss/checkpoints/model/sg_multimodal_v2_triloss/best.ckpt'
        #ckpt_path = '/u/wusifan/cadllm/results/eval_vitmae/best.ckpt'
        ckpt_path = '/Tmp/sifan/cad/best.ckpt'
        trainer.validate(model, ckpt_path=ckpt_path, dataloaders=sketchdata.test_dataloader())
    '''  
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
    '''
if __name__ == "__main__":
    main()
