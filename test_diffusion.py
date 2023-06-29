from io import BytesIO

import torch

import PIL
import requests
from diffusers import RePaintPipeline, RePaintScheduler


def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")


# img_url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/repaint/celeba_hq_256.png"
# mask_url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/repaint/mask_256.png"

# Load the original image and the mask as PIL images
original_image = PIL.Image.open("ori1.png").resize((256, 256))
mask_image = PIL.Image.open("test.png").resize((256, 256))

# Load the RePaint scheduler and pipeline based on a pretrained DDPM model
scheduler = RePaintScheduler.from_pretrained("google/ddpm-ema-celebahq-256")
pipe = RePaintPipeline.from_pretrained("google/ddpm-ema-celebahq-256", scheduler=scheduler)
pipe = pipe.to("cuda")

generator = torch.Generator(device="cuda").manual_seed(0)
output = pipe(
    image=original_image,
    mask_image=mask_image,
    num_inference_steps=250,
    eta=0.0,
    jump_length=10,
    jump_n_sample=10,
    generator=generator,
)
inpainted_image = output.images[0]