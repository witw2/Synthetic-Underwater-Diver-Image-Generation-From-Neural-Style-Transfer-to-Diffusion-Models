# src/model.py
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from params import MODEL_NAME, USE_CUDA

def load_diffusion_model():
    """Load a pretrained diffusion model."""
    model = StableDiffusionImg2ImgPipeline.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    model.to("cuda" if USE_CUDA else "cpu")
    return model
