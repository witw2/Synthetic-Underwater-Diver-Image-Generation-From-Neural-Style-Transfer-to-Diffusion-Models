# src/utils.py
import os
import random

import torch
from PIL import Image
from rembg import remove
from sklearn.metrics.pairwise import cosine_similarity

from params import *


def save_image(image, filename):
    # Ensure the output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Save the image to the primary output directory
    image.save(os.path.join(OUTPUT_DIR, filename))


def remove_background(image):
    # Convert the image to a format that rembg can handle (e.g., RGBA mode)
    image = image.convert("RGBA")  # Use RGBA to handle transparency
    image = remove(image)  # Remove the background
    return image


def merge_images(background, diver):
    if not (background.size == diver.size):
        raise ValueError("Background and diver images must be the same size.")

    # Ensure both images are RGB
    background = background.convert("RGB")
    diver = diver.convert("RGBA")

    # Composite the diver onto the background
    combined = Image.composite(diver, background, diver)

    return combined


def overlay_images(background, overlay, opacity=0.5):
    # Ensure both images are in RGBA mode for alpha blending
    background = background.convert("RGBA")
    overlay = overlay.convert("RGBA")

    # Adjust the alpha channel of the overlay based on the desired opacity
    overlay_with_opacity = Image.new("RGBA", overlay.size)
    for x in range(overlay.width):
        for y in range(overlay.height):
            r, g, b, a = overlay.getpixel((x, y))
            overlay_with_opacity.putpixel((x, y), (r, g, b, int(a * opacity)))

    # Composite the overlay onto the background
    combined = Image.alpha_composite(background, overlay_with_opacity)

    return combined


def pick_color(color_min, color_max):
    r = random.randint(color_min[0], color_max[0])
    g = random.randint(color_min[1], color_max[1])
    b = random.randint(color_min[2], color_max[2])
    return (r, g, b)


def scale_image(input_image, scale_factor):
    input_image = input_image.convert("RGBA")
    original_width, original_height = input_image.size

    # Scale down the image by the scale factor
    scaled_width = int(original_width * scale_factor)
    scaled_height = int(original_height * scale_factor)
    scaled_image = input_image.resize((scaled_width, scaled_height), Image.Resampling.BOX)

    transparent_background = Image.new("RGBA", (original_width, original_height), (0, 0, 0, 0))
    position = ((original_width - scaled_width) // 2, original_height - scaled_height)
    transparent_background.paste(scaled_image, position, scaled_image)
    return transparent_background


def compute_similarity_score(generated_image, base_image, model, processor):
    # Preprocess both images
    inputs_generated = processor(images=generated_image, return_tensors="pt", do_center_crop=True, do_resize=True)
    inputs_base = processor(images=base_image, return_tensors="pt", do_center_crop=True, do_resize=True)

    # Extract features
    with torch.no_grad():
        generated_features = model.get_image_features(**inputs_generated)
        base_features = model.get_image_features(**inputs_base)

    # Normalize features
    generated_features = generated_features / generated_features.norm(dim=-1, keepdim=True)
    base_features = base_features / base_features.norm(dim=-1, keepdim=True)

    # Compute cosine similarity
    similarity = cosine_similarity(generated_features.cpu().numpy(), base_features.cpu().numpy())[0][0]
    return similarity


def dummy_checker(images, **kwargs):
    # Return images and a list of False values (no NSFW content detected)
    return images, [False] * len(images)
