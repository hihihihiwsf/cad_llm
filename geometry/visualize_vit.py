from transformers import ViTFeatureExtractor
import requests
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt


class Visualize_VIT():
    
    def __init__(self, model, model_name="facebook/vit-mae-base"):
    
        feature_extractor = ViTFeatureExtractor.from_pretrained("facebook/vit-mae-base")
        self.imagenet_mean = np.array(feature_extractor.image_mean)
        self.imagenet_std = np.array(feature_extractor.image_std)
        self.model = model
        
    def show_image(self, image, title=''):
        # image is [H, W, 3]
        assert image.shape[2] == 3
        plt.imshow(torch.clip((image * self.imagenet_std + self.imagenet_mean) * 255, 0, 255).int())
        plt.title(title, fontsize=16)
        plt.axis('off')
        return

    def visualize(self, pixel_values):
        # forward pass
        outputs = self.model(pixel_values)
        y = self.model.unpatchify(outputs.logits)
        y = torch.einsum('nchw->nhwc', y).detach().cpu()
        
        # visualize the mask
        mask = outputs.mask.detach()
        mask = mask.unsqueeze(-1).repeat(1, 1, self.model.config.patch_size**2 *3)  # (N, H*W, p*p*3)
        mask = self.model.unpatchify(mask)  # 1 is removing, 0 is keeping
        mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
        
        x = torch.einsum('nchw->nhwc', pixel_values).cpu()

        # masked image
        im_masked = x * (1 - mask)

        # MAE reconstruction pasted with visible patches
        im_paste = x * (1 - mask) + y * mask

        # make the plt figure larger
        plt.rcParams['figure.figsize'] = [24, 24]

        plt.subplot(1, 4, 1)
        self.show_image(x[0], "original")

        plt.subplot(1, 4, 2)
        self.show_image(im_masked[0], "masked")

        plt.subplot(1, 4, 3)
        self.show_image(y[0], "reconstruction")

        plt.subplot(1, 4, 4)
        self.show_image(im_paste[0], "reconstruction + visible")

        plt.show()
        plt.savefig('test_ret.png')