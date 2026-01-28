# src/main.py
from io import BytesIO

from PIL import Image

from data_manager import save_image
from model import load_diffusion_model
from trainer import run_diffusion_process
from transform import preprocess_image
from params import TEXT_CONDITIONING_PROMPT, INPUT_DIR, OUTPUT_DIR, DEBUG, IMAGE_SIZE, REMOVE_BG
from utils import generate_variations_from_rgb_image
import os, torch

def main():
    # Load model
    model = load_diffusion_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # iterate over images from input directory
    for i, file in enumerate(os.listdir(INPUT_DIR)):
        # Load image
        if file.endswith(".png") or file.endswith(".jpg"):
            image = Image.open(os.path.join(INPUT_DIR, file)).convert("RGB")
        else:
            continue
        preprocessed_image = preprocess_image(image, [IMAGE_SIZE, IMAGE_SIZE], remove_bg=REMOVE_BG , should_save=True)

        # Run diffusion process
        generated_image = run_diffusion_process(model, preprocessed_image)

        transformed_image = generated_image

        # Save the final output
        save_image(transformed_image, f"output_{i}.png")
        transformed_image.show()


if __name__ == "__main__":
    main()
