import torch

BACKGROUND_PROMPT = (
    "A serene dark underwater scene with few soft light beams filtering through dark blue water entering at the "
    "surface. The water has a smooth gradient of dark blue, with faint floating particles and no visible objects or "
    "creatures. The scene is calm and immersive, with soft transitions and subtle depth. Theres no bottom, "
    "just endless blue"
)

BACKGROUND_NEGATIVE_PROMPT = (
    "sharp edges, creatures, corals, rocks, objects, focal points, unnatural patterns, overexposed light, "
    "chaotic textures"
)

NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5


def generate_underwater_background(model,
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
