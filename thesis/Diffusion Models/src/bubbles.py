import torch

# Usage example
BACKGROUND_PROMPT = (
    "A realistic underwater bubble column rising gently underwater. The bubbles are spherical, semi-transparent, "
    "and varying slightly in size as they ascend. The bubbles gradually fade, becoming smaller and less visible as "
    "they ascend. The background is neutral with no distracting elements."

)

BACKGROUND_NEGATIVE_PROMPT = (
    "sharp edges, unnatural patterns, chaotic textures, overexposed areas, creatures, rocks, corals, debris, "
    "artificial shapes, harsh lighting"
)

NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5


def generate_bubbles(model,
                     prompt=BACKGROUND_PROMPT,
                     size=(512, 512),
                     negative_prompt=BACKGROUND_NEGATIVE_PROMPT,
                     num_inference_steps=NUM_INFERENCE_STEPS,
                     guidance_scale=GUIDANCE_SCALE):
    # Load the text-to-image pipeline
    pipe = model
    # Generate the image
    with torch.no_grad():
        background = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=size[1],
            width=size[0]
        ).images[0]

    return background
