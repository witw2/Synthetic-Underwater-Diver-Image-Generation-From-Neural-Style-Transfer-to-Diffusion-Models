# src/data_manager.py

import os
from PIL import Image

# Function to load an image from a given path
def load_image(image_path):
    # Open the image file
    image = Image.open(image_path)
    # Convert the image to RGB mode
    image = image.convert('RGB')
    return image

# Function to save an image to a given path
def save_image(image, output_path):
    # Save the image file
    image.save(output_path)

# Function to get all images from a given directory
def get_images_from_directory(directory_path):
    # Define the valid extensions for image files
    valid_extensions = ('.jpg', '.jpeg', '.png')
    # Get a list of all files in the directory with a valid extension
    images = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith(valid_extensions)]
    return images