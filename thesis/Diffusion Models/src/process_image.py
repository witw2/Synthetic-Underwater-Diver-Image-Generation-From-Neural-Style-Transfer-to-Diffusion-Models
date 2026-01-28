NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 8
STRENGTH = 0.2
IMG2IMG_PROMPT = "Underwater scene with heavy blue colors and textures"


def process_image(img2img_model,
                  image,
                  prompt=IMG2IMG_PROMPT,
                  strength=STRENGTH,
                  num_diffusion_steps=NUM_DIFFUSION_STEPS,
                  guidance_scale=GUIDANCE_SCALE):
    img2img_model(prompt=prompt, image=image, num_diffusion_steps=num_diffusion_steps, guidance_scale=guidance_scale,
                  strength=strength).images[0]
    return image
