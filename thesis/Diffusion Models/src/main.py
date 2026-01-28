# src/main.py
import os
from random import randint
from time import time

import torch
import winsound
from PIL import Image, ImageDraw, ImageFont
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from transformers import CLIPModel, CLIPProcessor

from params import *
from src.process_image import process_image
from textures import generate_texture
from underwater_background import generate_underwater_background
from utils import save_image, remove_background, merge_images, overlay_images, pick_color, scale_image, \
    compute_similarity_score, dummy_checker

'''
workflow:
get surface photo
introduce variation to diver (optional)
remove background
generate underwater background
overlay underwater background
add underwater texture and tint to diver
'''


def main():
    # Load models
    img2img_model = StableDiffusionImg2ImgPipeline.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    img2img_model.safety_checker = dummy_checker
    img2img_model = img2img_model.to("cuda" if USE_CUDA else "cpu")

    txt2img_model = StableDiffusionPipeline.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    txt2img_model.safety_checker = dummy_checker
    txt2img_model = txt2img_model.to("cuda" if USE_CUDA else "cpu")

    extraction_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor_model = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # iterate over images from input directory
    for i, file in enumerate(os.listdir(INPUT_DIR)):
        # Load image
        if file.endswith(".png") or file.endswith(".jpg"):
            image = Image.open(os.path.join(INPUT_DIR, file)).convert("RGB")
        else:
            continue

        # Preprocess image
        image = image.resize(IMAGE_SIZE)

        # image = process_image(img2img_model, image)
        # Remove background
        image = remove_background(image)

        if SCALE_FACTOR != 1:
            image = scale_image(image, SCALE_FACTOR)

        score = 0
        # load random image from underwater folder
        base_image = Image.open(
            os.path.join(BASE_PATH, os.listdir(BASE_PATH)[randint(0, len(os.listdir(BASE_PATH)) - 1)])).convert("RGB")

        if DEBUG:
            draw = ImageDraw.Draw(base_image)
            font = ImageFont.truetype("arial.ttf", 50)
            draw.text((10, 10), "Base", fill="white", font=font)
            base_image.show()
        underwater_image = None
        start = time()
        j = 0
        best = None
        best_score = 0
        while score < SCORE and j < 15:
            j += 1
            # Generate underwater background
            underwater_image = generate_underwater_background(model=txt2img_model, size=IMAGE_SIZE)
            # Compute similarity score
            score = compute_similarity_score(underwater_image, base_image, extraction_model, processor_model)
            if score > best_score:
                best = underwater_image
                best_score = score
            if DEBUG:
                draw = ImageDraw.Draw(underwater_image)
                draw.text((10, 10), str(score), fill="white", font=font)
                underwater_image.show()
        underwater_image = best

        print(f"Underwater image generated in {time() - start} seconds with {j} iterations, score: {best_score}")

        # Overlay underwater background
        blended = merge_images(underwater_image, image)
        if INTRODUCE_VARIATION:
            blended = process_image(img2img_model, blended)

        # Generate texture
        texture = generate_texture(txt2img_model, size=IMAGE_SIZE)
        texture = overlay_images(blended, texture, opacity=0.3)

        # Overlay colour
        color = Image.new("RGB", IMAGE_SIZE, pick_color(COLOR_MIN, COLOR_MAX))
        color = overlay_images(texture, color, opacity=0.3)

        # Save image
        save_image(color, f"{file.split('.')[0]}_underwater.png")
        print(f"Image {file} processed")
        
    #play sound when done
    winsound.PlaySound("SystemQuestion", winsound.SND_ALIAS)


if __name__ == "__main__":
    main()
