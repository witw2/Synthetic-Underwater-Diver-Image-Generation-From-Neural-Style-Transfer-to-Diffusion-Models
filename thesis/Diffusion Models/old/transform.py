# src/transform.py
import random
import torch
from PIL import ImageDraw
from torchvision import transforms
from rembg import remove
from PIL import Image, ImageOps
import io
import os
import cv2
import numpy as np
import json

# Function to pick a color for the image
def pick_color(c="w"):
    if c=="w": #white for bubbles
        color = (np.random.randint(130, 201), np.random.randint(180, 231),np.random.randint(200, 256))
    elif c=="b": #blue for background
        color = (np.random.randint(33, 92),np.random.randint(116, 215),np.random.randint(182, 248))
    return color

def preprocess_image(image, target_size, remove_bg=False, should_save=False):
    # If the background needs to be removed
    if remove_bg:
        # Convert the image to a format that rembg can handle (e.g., RGBA mode)
        image = image.convert("RGBA")  # Use RGBA to handle transparency

        # Create a blue image of the same size as the original image
        blue_image = Image.new("RGBA", image.size, (14, 33, 50))

        # Blend the original image with the blue image
        image = Image.blend(image, blue_image, alpha=0.3)

        # Remove the background
        image_data = remove(image)

        # Convert the Image object to a bytes-like object
        byte_arr = io.BytesIO()
        image_data.save(byte_arr, format='PNG')
        byte_arr = byte_arr.getvalue()

        # Convert the background-removed image data back to a PIL image
        image = Image.open(io.BytesIO(byte_arr)).convert("RGBA")

        # Create a blue background image
        blue_bg = Image.new("RGBA", image.size, (49, 152, 231, 1))

        draw = ImageDraw.Draw(blue_bg)
        np.random.seed(sum(np.array(image).flatten()))

        non_zero_pixels = np.nonzero(np.array(image))
        min_y, max_y = np.min(non_zero_pixels[0]), np.max(non_zero_pixels[0])
        min_x, max_x = np.min(non_zero_pixels[1]), np.max(non_zero_pixels[1])
        addy=((max_y-min_y)*0.2)
        addx=((max_x-min_x)*0.2)
        bubble_size = image.width*0.0005
        # Draw spots on the image above diver to simulate air bubbles
        for _ in range(20):  # Number of spots
            x = np.random.randint(min_x+addx, max_x-addx)
            y = np.random.randint(0, min_y+addy)  # Draw above the object
            r = np.random.randint(10*bubble_size, 40*bubble_size)  # Radius of spots
            color = pick_color("w")
            draw.ellipse((x - r, y - r, x + r, y + r), fill=color)


        # Draw spots on the image to simmulate variations in the water
        for _ in range(40):  # Number of spots
            x = np.random.randint(0, image.width)
            y = np.random.randint(0, image.height)
            r = np.random.randint(5*bubble_size, 20*bubble_size)  # Radius of spots
            color = pick_color("w")
            draw.ellipse((x - r, y - r, x + r, y + r), fill=color)

        # Composite the original image onto the blue background
        image = Image.alpha_composite(blue_bg, image).convert("RGB")

        # Convert the image to RGB mode
        image = image.convert("RGB")

        # Convert PIL Image to OpenCV format
        open_cv_image = np.array(image)
        open_cv_image = open_cv_image[:, :, ::-1].copy()

        # Create mask (this depends on your image)]
        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)

        # Blur image
        blurred = cv2.GaussianBlur(open_cv_image, (21, 21), 0)

        # Combine original and blurred image using the mask
        soft_edges = np.where(mask[..., None] == 255, blurred, open_cv_image)

        # Convert OpenCV image back to PIL format
        image = Image.fromarray(cv2.cvtColor(soft_edges, cv2.COLOR_BGR2RGB))

    else:
        # If not removing background, ensure the image is in RGB mode
        image = image.convert("RGB")

    # Resize the image to the target size
    image = image.resize(target_size)

    if should_save:
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.realpath(__file__))

        # Get the project directory (assuming the script is in the 'src' subdirectory of the project directory)
        project_dir = os.path.dirname(script_dir)

        # Ensure the directory exists
        preprocessed_dir = os.path.join(project_dir,'data', 'preprocessed')
        os.makedirs(preprocessed_dir, exist_ok=True)
        # Save the preprocessed image
        preprocessed_image_path = os.path.join(preprocessed_dir, 'preprocessed_image.jpg')
        image.save(preprocessed_image_path)

    return image