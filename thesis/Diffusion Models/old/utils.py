# src/utils.py
import torch
from diffusers import StableDiffusionImageVariationPipeline, StableDiffusionImg2ImgPipeline
from PIL import Image
from transformers import CLIPTokenizer
from torchvision import transforms
from model import load_diffusion_model

def truncate_prompt(prompt, max_length=77):
    if len(prompt) < max_length:
        return prompt
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    tokens = tokenizer(prompt, truncation=True, max_length=max_length, return_tensors="pt")
    truncated_prompt = tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)
    return truncated_prompt


def generate_variations_from_rgb_image(input_image, guidance_scale=3):
    if not isinstance(input_image, Image.Image):
        raise ValueError("Input image must be a PIL.Image.Image object.")

    # Load the variation pipeline
    pipe = StableDiffusionImageVariationPipeline.from_pretrained("lambdalabs/sd-image-variations-diffusers", revision="v2.0")
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    im = input_image
    tform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(
            (224, 224),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=False,
        ),
        transforms.Normalize(
            [0.48145466, 0.4578275, 0.40821073],
            [0.26862954, 0.26130258, 0.27577711]),
    ])
    inp = tform(im).to("cuda" if torch.cuda.is_available() else "cpu").unsqueeze(0)
    out = pipe(inp, guidance_scale=guidance_scale)

    return out["images"][0]