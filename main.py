"""
Train a CAD LLM model
"""


try:
    import comet_ml  # Import before torch
except ImportError:
    pass
# from dataset.sg_dataset_visrecon import get_sketchgraphs_dataloader
from dataset.sg_dataset import get_sketchgraphs_dataloader, get_icl_sketchgraphs_dataloader

from models import vl_byt5, byt5, vl_byt5_v2

from models.vis_recon import VisRecon
from torch.utils.data import DataLoader
from util import get_loggers, get_checkpoint_callbacks, EmbeddingCallback
from args.main_args import get_training_args
from pathlib import Path
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor
import os
from pytorch_lightning.strategies import DDPStrategy

from transformers import AutoTokenizer
import matplotlib.pyplot as plt

from geometry.parse import get_curves, get_point_entities
from geometry.visualization import visualize_batch, visualize_sample, visualize_sample_cv, visualize_sample_pil

from dataset.sg_dataset_savefeature import save_icl_sketchgraphs_dataloader

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
    
    print("Loading model...")

    architecture = vl_byt5_v2
    
    from transformers import ViTMAEForPreTraining 
    # vitmae_model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
    tokenizer = architecture.ByT5Model.get_tokenizer(args.model_name)

    print("Loading data...")
    train_dataloader = get_icl_sketchgraphs_dataloader(tokenizer=tokenizer, args=args, split="train", shuffle=True)
    
    val_dataloader = get_icl_sketchgraphs_dataloader(tokenizer=tokenizer, args=args, split="val", shuffle=False)
    test_dataloader = get_icl_sketchgraphs_dataloader(tokenizer=tokenizer, args=args, split="test", shuffle=False)

    num_train_batches = len(train_dataloader)
    num_gpus = torch.cuda.device_count()
    total_train_steps = architecture.ByT5Model.get_total_train_steps(num_train_batches, num_gpus, args.epochs)

    model = architecture.ByT5Model(args=args, tokenizer=tokenizer, total_train_steps=total_train_steps, vit_mae=None)
    
    call_backs = get_checkpoint_callbacks(log_dir=results_dir, all_checkpoint_dir=checkpoint_dir,
                                          using_sagemaker=args.using_sagemaker)

    call_backs.append(LearningRateMonitor(logging_interval='step'))
    
    # embedding_callback= EmbeddingCallback()
    

    print("Training the model...")
    log_every_n_steps = 50
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"

    trainer = pl.Trainer(
        callbacks=call_backs,   #[embedding_callback],
        accelerator= args.accelerator,
        devices=args.devices,
        strategy=DDPStrategy(find_unused_parameters=True),
        logger=loggers,
        max_epochs=args.epochs,
        log_every_n_steps=log_every_n_steps,
        # resume_from_checkpoint=None,
        precision='16-mixed',
        check_val_every_n_epoch=args.val_every_n_epoch,
        #limit_train_batches=0.1,
        #limit_val_batches=0.1,
    )
    if not args.eval: 
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        trainer.test(model, dataloader=test_dataloader, ckpt_path='best')
    else:
        # loading the model from exp_name/best.ckpt
        ckpt_dir = args.checkpoint_dir + "/{}/best.ckpt".format(args.exp_name)
        #path = 's3://cad-llm-katzm/jobs/amir-Codet5p-ascii1-max64-deepmindv1-06-09-23-2253/checkpoints/model/Codet5p_ascii1_max64_deepmindv1/best.ckpt'
        trainer.validate(model, ckpt_path=ckpt_dir, dataloaders=val_dataloader)
    
        print("end evaluation")
    
        
        # #saved_image_embeddings = embedding_callback.image_embeddings
        # saved_txt_embeddings = embedding_callback.txt_embeddings
        # saved_pixels = embedding_callback.pixel
        # saved_name = embedding_callback.name
        # saved_fulltext = embedding_callback.fulltext
        # saved_intext = embedding_callback.intext
        
        # #import pdb; pdb.set_trace()
        
        # import numpy as np
        # np.save('embeddings.npy', saved_txt_embeddings)
        # np.save('pixel.npy', saved_pixels)
        # np.save('intext.npy', saved_intext)
        # np.save('name.npy', saved_name)
        # np.save('fulltext.npy', saved_fulltext)
        
    '''
    import numpy as np
    from sklearn.manifold import TSNE
    
    import umap
    import faiss
    all_embeddings = np.concatenate(saved_embeddings, axis=0)
    all_pixels = np.concatenate(saved_pixels, axis=0)
    all_fulltext = np.concatenate(saved_fulltext, axis=0)
    all_intext = np.concatenate(saved_intext, axis=0)
    
    #all_embeddings = all_embeddings.reshape(all_embeddings.shape[0], -1)
    np.save('embeddings.npy', all_embeddings)
    np.save('pixel.npy', all_pixels)
    
    # tsne = TSNE(n_components=3, random_state=42)
    # tsne_embeddings = tsne.fit_transform(all_embeddings)
    
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(8, 6))
    # plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], marker='o')
    # plt.title('t-SNE Embeddings')
    # plt.savefig('tsne-valid-embeddings')
    
    # reducer = umap.UMAP(n_neighbors=200, n_components=3,n_epochs=1000)
    # umap_embeddings = reducer.fit_transform(all_embeddings)
    
    
    # from mpl_toolkits.mplot3d import Axes3D
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], umap_embeddings[:, 2])
    # plt.title('3D UMAP Scatter Plot')
    # plt.savefig('3D_UMAP.png')
    
    # faiss.normalize_L2(all_embeddings)
    # index = faiss.IndexFlatL2(768)
    # index.add(all_embeddings)
    
    topk = 5
    emb = torch.Tensor(all_embeddings[1,-1,:])
    emb = emb.repeat(all_embeddings.shape[0], 1)
    embeddings = torch.Tensor(all_embeddings[:,-1,:])
    cos = torch.nn.CosineSimilarity(dim=1,eps=1e-6)
    out = cos(emb, embeddings)
    idx = np.argsort(out)[:topk]
    similar_p = all_pixels[idx]
    
    p0 = all_pixels[0]
    pixels = np.clip(p0, 0, 1)
    pixels = np.transpose(pixels, (1, 2, 0))
    
    plt.figure(figsize=(10, 10))
    plt.subplot(3,2,1)
    plt.imshow(pixels)
    for i in range(topk):
        image = similar_p[i]
        pixel = np.clip(image, 0, 1)
        pixel = np. transpose(pixel, (1,2,0))
        
        plt.subplot(3,2,i+2)
        plt.imshow(pixel)
    plt.savefig('')
    '''
        
      
        
if __name__ == "__main__":
    # print("test")    
    
    
    
    from line_profiler import LineProfiler
    lp = LineProfiler()
    
    main()
    lp.print_stats()
