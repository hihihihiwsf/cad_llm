"""
ByT5 pytorch lightning model
"""

try:
    import comet_ml  # Import before torch
except ImportError:
    pass
import torch
import torch.optim as optim
# import lightning.pytorch as pl
import pytorch_lightning as pl
from transformers import T5Config, AutoTokenizer
from transformers.modeling_utils import unwrap_model
import sys

sys.path.insert(0, '/home/ec2-user/SageMaker/efs/code/cad_llm')
from metrics import calculate_accuracy, calculate_first_ent_accuracy, calculate_validity
from util import get_quantized_range
from geometry.parse import get_curves, get_point_entities
from geometry.visualization import visualize_batch, visualize_sample
from pathlib import Path
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from PIL import Image
# import clip
import numpy as np
# from transformers import CLIPVisionModelWithProjection, CLIPVisionModel, ViTMAEModel, ViTMAEForPreTraining, ViTMAEConfig
from models.modeling_vit_mae import ViTMAEModel, ViTMAEForPreTraining, ViTMAEConfig
from geometry.visualize_vit import Visualize_VIT


class VisRecon(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()

    
        self.model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")

        
        # self.model.requires_grad_(False)
        # self.m = torch.load('checkpoints/vitmae_deepmind/best.ckpt')
        
        self.args = args

        self.lr = self.args.lr
        self.batch_size = self.args.batch_size  # to fix logging warning
        self.quantization_bits = 6  # Hard code for now
        self.quantized_range = get_quantized_range(self.quantization_bits)
        self.box_lim = max(self.quantized_range)  # for visualization
        self.vis_vit = Visualize_VIT(self.model)
        
        
        # self.clip_model, _ = clip.load(args.clipmodel)
        # self.clip_model = CLIPVisionModelWithProjection.from_pretrained(args.clipmodel)
        # self.clip_model = CLIPVisionModel.from_pretrained(args.clipmodel, output_hidden_states=True)
        # self.clip_model.requires_grad_(False)


        
        # self.mapper =torch.nn.Linear(self.clip_model.visual.output_dim, self.model.get_input_embeddings().weight.shape[1])
        # self.mapper =torch.nn.Linear(self.clip_model.config.projection_dim, self.model.get_input_embeddings().weight.shape[1])
        # self.mapper =torch.nn.Linear(self.clip_model.config.hidden_size, self.model.get_input_embeddings().weight.shape[1])



        # If using single token encoding - adjust tokenizer and model embeddings
        if not args.ascii_encoding:
            self.adjust_to_use_new_tokens()

        if args.lora:
            self.add_lora()

    def training_step(self, batch, batch_idx):



        outputs = self.model(**batch['images'])
        loss = outputs.loss  # CrossEntropyLoss(ignore_index=-100) between outputs.logits and labels
        
        # px = batch['images']
        # px = px['pixel_values']
        # self.vis_vit.visualize(torch.unsqueeze(px[0], 0).to(self.model.device))
        
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True,
                 batch_size=self.batch_size, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        


        outputs = self.model(**batch['images'])
        loss = outputs.loss
        # if self.current_epoch == 0:
            # print("epoch 0 begin draw image")
        px = batch['images']
        px = px['pixel_values']
        self.vis_vit.visualize(torch.unsqueeze(px[0], 0).to(self.model.device))
        #self.vis_vit.visualize(px.to(self.model.device))
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=self.batch_size, sync_dist=True)



    def generate_samples(self, batch):
        # Recursively unwrap the model from potential distributed training containers
        generate_func = unwrap_model(self.model).generate
        batch["samples"] = generate_func(inputs_embeds=batch["inputs_embeds"], attention_mask=batch["attention_mask"],
                                         do_sample=False, max_new_tokens=self.args.max_length+10)

        batch["string_samples"] = self.tokenizer.batch_decode(batch["samples"], skip_special_tokens=True)
        batch["string_labels"] = [sketch["output_text"] for sketch in batch["sketches"]]

        batch["point_samples"] = [get_point_entities(string_sample) for string_sample in batch["string_samples"]]
        batch["point_labels"] = [get_point_entities(string_label) for string_label in batch["string_labels"]]

        batch["sample_curves"] = [get_curves(point_sample) for point_sample in batch["point_samples"]]

    def log_samples(self, batch, batch_idx):
        label_curves = [get_curves(point_label) for point_label in batch["point_labels"]]

        point_inputs = [get_point_entities(sketch["input_text"]) for sketch in batch["sketches"]]
        input_curves = [get_curves(point_input) for point_input in point_inputs]

        fig = visualize_batch(input_curves=input_curves, label_curves=label_curves,
                              sample_curves=batch["sample_curves"], box_lim=self.box_lim + 3)
        fig_path = Path(self.args.samples_dir) / f"epoch_{self.current_epoch}_batch_{batch_idx}.png"
        fig.savefig(fig_path)

    def configure_optimizers(self):
        # params = list(self.model.parameters()) + list(self.mapper.parameters())
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        if not self.args.cosinedecay:
            return optimizer
            
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(self.args.epochs * 1.15), verbose=True)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=4,sefsdfsdf verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        }