from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import json
from pathlib import Path
from geometry.parse import get_curves, get_point_entities
from geometry.visualization import visualize_batch, visualize_sample, visualize_sample_cv, visualize_sample_pil
# import clip
import torch 
from transformers import CLIPImageProcessor, AutoImageProcessor, ViTMAEModel


class sketchGraphRetrievalDataset(Dataset):
    def __init__(self, args, split):
        path = Path(args.retrieved_dataset) / f"{split}.json"
        with open(path, "r") as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        """
        Applies a random mask to the entities of sketch
        Returns (input_text, output_text, retrieved_text)
        """
        sketch_dict = self.data[index]

        # sketch_dict['input_text'] = input_text
        # sketch_dict['output_text'] = output_text
        sketch_dict['icl_text'] = sketch_dict['prompt'].split('\n')[0].replace('\t','')
        return sketch_dict
   
class SketchGraphsRetrievalCollator:
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
        input_strings = [sketch['input_text'] for sketch in sketch_dicts]
        output_strings = [sketch['output_text'] for sketch in sketch_dicts]
        tokenized_input = self.tokenize(input_strings)
        tokenized_output = self.tokenize(output_strings)

        labels = tokenized_output.input_ids
        # replace padding token id's of the labels by ignore_index=-100 so it's ignored by the loss
        labels[labels == self.tokenizer.pad_token_id] = -100

        point_inputs = [get_point_entities(sketch["input_text"]) for sketch in sketch_dicts]
        list_of_img = visualize_sample_pil(point_entities=point_inputs, box_lim=64 + 3)
        input_batch_images = self.vitmae_preprocess(list_of_img, return_tensors="pt")
        input_batch_images = input_batch_images
        
        
        point_inputs = [get_point_entities(sketch["icl_text"]) for sketch in sketch_dicts]
        list_of_img = visualize_sample_pil(point_entities=point_inputs, box_lim=64 + 3)
        icl_batch_images = self.vitmae_preprocess(list_of_img, return_tensors="pt")
        icl_batch_images = icl_batch_images
        
        # images = []
        # for img in list_of_img:
        #     images.append(self.clip_preprocess(img))
        #     batch_images = torch.tensor(np.stack(images))

        # for im in list_of_img:
        #     im.close()
        
              
        batch = {
            "input_ids": tokenized_input.input_ids,
            "attention_mask": tokenized_input.attention_mask,
            "labels": labels,
            "sketches": sketch_dicts,
            "input_images": input_batch_images,
            "icl_image": icl_batch_images,
        }
        return batch 

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
        
        self.token_sum = 0

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
        input_text = "".join([ent for i, ent in enumerate(entities) if mask[i]])
        output_text = "".join([ent for i, ent in enumerate(entities) if not mask[i]])
        sketch_dict['input_text'] = input_text
        sketch_dict['output_text'] = output_text
        sketch_dict['full_text'] = input_text+output_text
        self.token_sum += len(sketch_dict['full_text'])
        
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
        return self.token_sum #len(self.data)


class SketchGraphsCollator:
    def __init__(self, tokenizer, max_length=None, args=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.args = args
        # _, self.clip_preprocess = clip.load("ViT-B/32")
        # self.clip_preprocess = CLIPImageProcessor.from_pretrained(self.args.clipmodel)
        self.vitmae_preprocess = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")

    def tokenize(self, strings):
        return self.tokenizer(strings, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")

    def __call__(self, sketch_dicts):
        input_strings = [sketch['input_text'] for sketch in sketch_dicts]
        output_strings = [sketch['output_text'] for sketch in sketch_dicts]
        tokenized_input = self.tokenize(input_strings)
        tokenized_output = self.tokenize(output_strings)

        labels = tokenized_output.input_ids
        # replace padding token id's of the labels by ignore_index=-100 so it's ignored by the loss
        labels[labels == self.tokenizer.pad_token_id] = -100

        point_inputs = [get_point_entities(sketch["input_text"]) for sketch in sketch_dicts]
        # input_curves = [get_curves(point_input) for point_input in point_inputs]
        # list_of_img = visualize_sample(input_curves=input_curves, box_lim=64 + 3)
        list_of_img = visualize_sample_pil(point_entities=point_inputs, box_lim=64 + 3)

        # batch_images = self.clip_preprocess(list_of_img, return_tensors="pt")
        batch_images = self.vitmae_preprocess(list_of_img, return_tensors="pt")
        #batch_images['pixel_values'] = torch.zeros_like(batch_images['pixel_values'])
        # images = []
        # for img in list_of_img:
        #     images.append(self.clip_preprocess(img))
        #     batch_images = torch.tensor(np.stack(images))

        # for im in list_of_img:
        #     im.close()
        
              
        batch = {
            "input_ids": tokenized_input.input_ids,
            "attention_mask": tokenized_input.attention_mask,
            "labels": labels,
            "sketches": sketch_dicts,
            "images": batch_images,
        }
        return batch


def get_sketchgraphs_dataloader(tokenizer, args, split, shuffle):
    
    dataset = SketchGraphsDataset(split=split, args=args)
    
    collator = SketchGraphsCollator(tokenizer=tokenizer, max_length=args.max_length, args=args)
    return DataLoader(dataset, batch_size=args.batch_size, collate_fn=collator, shuffle=shuffle,
                      num_workers=args.num_workers)


def get_icl_sketchgraphs_dataloader(tokenizer, args, split, shuffle):
    dataset = sketchGraphRetrievalDataset(split=split, args=args)
    collator = SketchGraphsRetrievalCollator(tokenizer=tokenizer, max_length=args.max_length, args=args)
    return DataLoader(dataset, batch_size=args.batch_size, collate_fn=collator, shuffle=shuffle,
                      num_workers=args.num_workers)
