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
from transformers import T5Config, T5ForConditionalGeneration, AutoTokenizer
from transformers.modeling_utils import unwrap_model
import sys

sys.path.insert(0, '/home/ec2-user/SageMaker/efs/code/cad_llm')
from metrics import calculate_f1, calculate_accuracy, calculate_first_ent_accuracy, calculate_validity
from util import get_quantized_range
from geometry.parse import get_curves, get_point_entities
from geometry.visualization import visualize_batch
from pathlib import Path
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.vis_recon import VisRecon
from transformers import AutoImageProcessor

class ByT5Model(pl.LightningModule):
    total_train_steps = None
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()

        if args.untrained_model:
            config = T5Config.from_pretrained(args.model_name)
            model = T5ForConditionalGeneration(config)
            model._init_weights(model)  # maybe redundant
        else:
            model = T5ForConditionalGeneration.from_pretrained(args.model_name)

        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        self.args = args

        self.lr = self.args.lr
        self.batch_size = self.args.batch_size  # to fix logging warning
        self.total_train_steps = None  # should be set later for lr scheduler

        self.quantization_bits = 6  # Hard code for now
        self.quantized_range = get_quantized_range(self.quantization_bits)
        self.box_lim = max(self.quantized_range)  # for visualization
        
        '''image input part'''
        m = VisRecon(args=args)
        m.load_from_checkpoint('~/Projects/cad_llm/checkpoints/vitmae_deepmind/best.ckpt')
        self.vit_mae = m.model
        del m
        
        self.mapper =torch.nn.Linear(self.vis_model.config.hidden_size, self.model.get_input_embeddings().weight.shape[1])
        self.post_layernorm = torch.nn.LayerNorm(self.vis_model.config.hidden_size, eps=1e-5)
        self.layernorm = torch.nn.LayerNorm(self.model.get_input_embeddings().weight.shape[1], eps=1e-5)

        self.patch_num = int(self.vis_model.config.image_size/self.vis_model.config.patch_size)

        self.embed_patch = torch.nn.Linear(self.patch_num*self.patch_num, self.patch_num)
        self.gelu = torch.nn.GELU()
        

    def training_step(self, batch, batch_idx):
        cols = ["labels"]
        model_batch = {col: val for col, val in batch.items() if col in cols}
        
        with torch.no_grad():
            oi = self.vis_model.vit.encoder(self.vis_model.patchify(**batch['images']))
            image_features = torch.sum(oi['last_hidden_state'], 1)
        
        image_embeds = self.post_layernorm(oi['last_hidden_state'])
        image_embeds = image_embeds.permute(0,2,1)
        image_embeds = self.gelu(self.embed_patch(image_embeds).permute(0,2,1))
        image_for_llm = self.layernorm(self.gelu(self.mapper(image_embeds)))
        
        txt_embedder = self.model.get_input_embeddings()
        txt_embeddings = txt_embedder(batch['input_ids'])
        
        input_embeddings = self.layernorm(torch.concat((image_for_llm, txt_embeddings), 1))
        model_batch['inputs_embeds'] = input_embeddings
        
        att = batch['attention_mask']
        model_batch['attention_mask'] = torch.cat((torch.ones(att.shape[0], image_for_llm.shape[1]).to(self.device), att), dim=1)
        
        outputs = self.model(**model_batch)
        loss = outputs.loss  # CrossEntropyLoss(ignore_index=-100) between outputs.logits and labels
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True,
                 batch_size=self.batch_size, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        cols = ["input_ids", "attention_mask", "labels"]
        model_batch = {col: val for col, val in batch.items() if col in cols}
        outputs = self.model(**model_batch)
        loss = outputs.loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=self.batch_size, sync_dist=True)

        # Generate and process samples
        self.generate_samples(batch)

        # Calculate metrics
        f1 = calculate_f1(samples=batch["point_samples"], labels=batch["point_labels"])
        self.log("f1", f1, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=self.batch_size, sync_dist=True)
	
        top1_full_sketch = calculate_accuracy(samples=batch["point_samples"], labels=batch["point_labels"])
        self.log("top1_full_sketch", top1_full_sketch, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=self.batch_size, sync_dist=True)

        top1_ent = calculate_first_ent_accuracy(samples=batch["point_samples"], labels=batch["point_labels"])
        self.log("top1_ent", top1_ent, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=self.batch_size, sync_dist=True)

        # Convert string entities to curves and check validity
        validity = calculate_validity(batch_sample_curves=batch["sample_curves"])
        self.log("validity", validity, on_step=False, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=self.batch_size, sync_dist=True)

        # # Plot sketches
        if batch_idx < 5:
            self.log_samples(batch=batch, batch_idx=batch_idx)

        return loss

    def generate_samples(self, batch):
        # Recursively unwrap the model from potential distributed training containers
        generate_func = unwrap_model(self.model).generate
        batch["samples"] = generate_func(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                         do_sample=False, max_new_tokens=self.args.max_length+10)

        string_samples = self.tokenizer.batch_decode(batch["samples"], skip_special_tokens=True)
        batch["point_samples"] = [get_point_entities(string_sample) for string_sample in string_samples]
        batch["point_labels"] = [get_point_entities(string_label) for string_label in batch["output_text"]]

        batch["sample_curves"] = [get_curves(point_sample) for point_sample in batch["point_samples"]]

    def log_samples(self, batch, batch_idx):
        label_curves = [get_curves(point_label) for point_label in batch["point_labels"]]

        batch["point_inputs"] = [get_point_entities(string_label) for string_label in batch["input_text"]]
        input_curves = [get_curves(point_input) for point_input in batch["point_inputs"]]

        fig = visualize_batch(input_curves=input_curves, label_curves=label_curves,
                              sample_curves=batch["sample_curves"], box_lim=self.box_lim + 3)
        fig_path = Path(self.args.samples_dir) / f"epoch_{self.current_epoch}_batch_{batch_idx}.png"
        fig.savefig(fig_path)

    def set_total_train_steps(self, num_train_batches):
        # Assumes running on gpus, one node and no accumulate_grad_batches
        n_gpus = torch.cuda.device_count()
        train_batches = num_train_batches // n_gpus if n_gpus else num_train_batches
        self.total_train_steps = train_batches * self.args.epochs
    
    @staticmethod
    def set_total_train_steps_ray(num_train_batches, n_gpus, epochs):
        # Assumes running on gpus, one node and no accumulate_grad_batches
        n_gpus = torch.cuda.device_count()
        train_batches = num_train_batches // n_gpus if n_gpus else num_train_batches
        ByT5Model.total_train_steps = train_batches * epochs

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        if not self.args.cosinedecay:
            return optimizer

        scheduler = CosineAnnealingLR(optimizer, T_max=self.total_train_steps, eta_min=self.lr * 0.1)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=4, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }
