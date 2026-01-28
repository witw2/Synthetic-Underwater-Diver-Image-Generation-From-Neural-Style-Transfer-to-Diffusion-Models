from PIL import ImageEnhance

PROMPT = ("An abstract underwater distortion texture with ripples and particles. Avoid sharp edges and details. Add a "
          "slight blue tint. Floating particles are fine. Background should be greenish-blue. Texture have to be "
          "subtle."
          )
GUIDANCE_SCALE = 7.5
TRANSPARENCY = 0.3


def generate_texture(model,
                     prompt=PROMPT,
                     size=(512, 512),
                     guidance_scale=7.5,
                     strength=0.1,
                     transparency=0.3):
    # Generate texture using a text-to-image model
    texture = model(prompt, guidance_scale=guidance_scale, height=size[1], width=size[0]).images[0]
    # make the texture transparent
    enhancer = ImageEnhance.Brightness(texture)
    texture = enhancer.enhance(strength)
    texture.putalpha(int(255 * transparency))

    return texture
