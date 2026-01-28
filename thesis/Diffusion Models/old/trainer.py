# src/trainer.py
import torch
from params import (NUM_DIFFUSION_STEPS, TEXT_CONDITIONING_PROMPT, GUIDANCE_SCALE, STRENGTH, NEGATIVE_PROMPT)
from utils import truncate_prompt

def run_diffusion_process(model, input_image):
    """Runs the diffusion process on an input image using a prompt."""
    truncated_prompt = truncate_prompt(TEXT_CONDITIONING_PROMPT)
    truncated_negative_prompt = truncate_prompt(NEGATIVE_PROMPT)
    with torch.no_grad():
        output = model(
            prompt=truncated_prompt, # Text prompt to guide the transformation
            negative_prompt=truncated_negative_prompt, # Negative text prompt to avoid
            image=input_image,                     # Input image
            strength=STRENGTH,                    # Control the level of transformation
            num_inference_steps=NUM_DIFFUSION_STEPS,
            guidance_scale=GUIDANCE_SCALE       # Control the influence of the prompt
        ).images
    return output[0]