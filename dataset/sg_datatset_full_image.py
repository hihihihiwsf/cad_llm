from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import json
from pathlib import Path
from geometry.parse import get_curves, get_point_entities
from geometry.visualization import visualize_batch, visualize_sample
# import clip
import torch 
from transformers import CLIPImageProcessor, AutoImageProcessor, ViTMAEModel
import multiprocessing as mp

def get_ids_restore(full_pixel, masked_pixel):
    k=0
    keep_len=0
    bsz, seq_length, dim = full_pixel.shape
    batch_mask = np.zero((bsz,seq_length))
    masks_ = np.ones((bsz,seq_length))
    batch_ids_restore = np.rand(bsz, seq_length)
    for k, (full, masked) in enumerate(zip(full_pixel, masked_pixel)):
        dif = torch.eq(full, masked)
        for i in seq_length:
            if torch.all(dif[i]):
                batch_mask[k][i] = 0
                keep_len = 0
            else:
                batch_mask[k][i]=1
    masks_[:,:keep_len] = 0
    batch_ids_restore = (masks_.unsqueeze(0) == batch_mask.unsqueeze(1)).nonzero()[:, -2]
    
    return batch_ids_restore, batch_mask

class SketchGraphsDataset(Dataset):
    def __init__(self, args, split):
        path = Path(args.dataset) / f"{split}.json"
        with open(path, "r") as f:
            self.data = json.load(f)

        if split == "train":
            n = int(args.limit_data * len(self.data))
            random.shuffle(self.data)
            self.data = self.data[:n]

        self.order = args.train_order if split == "train" else "sorted"
        assert self.order in ["sorted", "user", "random"]
        self.entities_col = "user_ordered_entities" if self.order == "user" else "entities"

        # Sanity check text format
        entity_string = self.data[0][self.entities_col][0]
        if args.ascii_encoding:
            error_message = f"Expected format '-31, 17' not '<-31><17>', found '{entity_string}'"
            assert entity_string[0] != "<", error_message
            assert "," in self.data[0][self.entities_col][0], error_message
        else:
            error_message = f"Expected format '<-31><17>' not '-31, 17', found '{entity_string}'"
            assert entity_string[0] == "<", error_message
            assert "," not in self.data[0][self.entities_col][0], error_message

        self.min_input_percent = args.min_input_percent
        self.max_input_percent = args.max_input_percent
        assert self.min_input_percent >= 0 and self.max_input_percent <= 1

    def __getitem__(self, index):
        """
        Applies a random mask to the entities of sketch
        Returns (input_text, output_text)
        """
        sketch_dict = self.data[index]
        entities = sketch_dict[self.entities_col]
        if self.order == "random":
            np.random.shuffle(entities)
        mask = self.get_mask(len(entities))
        sketch_dict["mask"] = mask
        full_text = "".join([ent for i, ent in enumerate(entities)])
        input_text = "".join([ent for i, ent in enumerate(entities) if mask[i]])
        output_text = "".join([ent for i, ent in enumerate(entities) if not mask[i]]) #if not mask[i]
        sketch_dict['input_text'] = input_text
        sketch_dict['output_text'] = output_text
        sketch_dict['full_text'] = full_text

        return sketch_dict

    def get_mask(self, n):
        """
        Sample a random size for mask and a random mask of size n
        """
        mask_percent = random.uniform(self.min_input_percent, self.max_input_percent)
        mask_size = round(mask_percent * n)
        mask_size = min(max(1, mask_size), n - 1)

        mask = np.zeros(n, dtype=bool)
        mask[:mask_size] = 1
        np.random.shuffle(mask)
        return mask

    def __len__(self):
        return len(self.data)


class SketchGraphsCollator:
    def __init__(self, tokenizer, max_length=None, args=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.args = args
        # _, self.clip_preprocess = clip.load("ViT-B/32")
        # self.clip_preprocess = CLIPImageProcessor.from_pretrained(self.args.clipmodel)
        self.vitmae_preprocess = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")

    def tokenize(self, strings):
        return self.tokenizer(strings, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")

    def __call__(self, sketch_dicts):

        point_inputs = [get_point_entities(sketch["full_text"]) for sketch in sketch_dicts]
        input_curves = [get_curves(point_input) for point_input in point_inputs]
        
        # proc = mp.Process(target=visualize_sample(input_curves=input_curves, box_lim=64 + 3))
        list_of_img = visualize_sample(input_curves=input_curves, box_lim=64 + 3)
        # proc.daemon=True
        # proc.start()
        # proc.join()

        # batch_images = self.clip_preprocess(list_of_img, return_tensors="pt")
        batch_pixel_values = self.vitmae_preprocess(list_of_img, return_tensors="pt")

        point_inputs = [get_point_entities(sketch["input_text"]) for sketch in sketch_dicts]
        input_curves = [get_curves(point_input) for point_input in point_inputs]
        
        list_of_img = visualize_sample(input_curves=input_curves, box_lim=64 + 3)
        
        batch_masked_values = self.vitmae_preprocess(list_of_img, return_tensors="pt")

        batch_ids_restore, batch_masks = get_ids_restore(batch_pixel_values, batch_masked_values)
        batch = {
            "sketches": sketch_dicts,
            "pixel_values": batch_pixel_values,
            "masked_values": batch_masked_values,
            "ids_restore": batch_ids_restore,
            "batch_mask": batch_masks
        }
        return batch


def get_sketchgraphs_dataloader(tokenizer, args, split, shuffle):
    dataset = SketchGraphsDataset(split=split, args=args)
    collator = SketchGraphsCollator(tokenizer=tokenizer, max_length=args.max_length, args=args)
    return DataLoader(dataset, batch_size=args.batch_size, collate_fn=collator, shuffle=shuffle,
                      num_workers=args.num_workers)
