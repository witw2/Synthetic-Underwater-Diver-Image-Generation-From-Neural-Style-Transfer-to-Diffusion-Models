import os
import winsound

from src.data_manager import get_images_from_directory
from src.utils import combine_styles
from src.optimizer import optimize_parameters

def main_optimize():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.realpath(__file__))
    # Define the directories for input and style images
    input_dir = os.path.join(script_dir, 'data/input/')
    style_dir = os.path.join(script_dir, 'data/style/1/')

    # Get all images from the input and style directories
    input_images = get_images_from_directory(input_dir)
    style_images = get_images_from_directory(style_dir)

    # Combine all style images into one
    combined_style_image = combine_styles(style_images)
    # Save the combined style image
    combined_style_image.save(f"{os.path.join(script_dir,'data/preprocessed')}/combined_style.jpg")

    # Define the number of steps for the optimizer
    num_steps = 10000

    # Iterate over all input images
    for image_path in input_images:
        # Optimize the parameters for each input image with the combined style image
        best_params = optimize_parameters(image_path, combined_style_image, num_steps,50)
        # Print the best parameters for each input image
        print(f"Best parameters for {os.path.basename(image_path)}: {best_params}")

if __name__ == "__main__":
    # Run the main optimize function
    main_optimize()
    # Play a sound when the optimization is done
    winsound.PlaySound("SystemQuestion", winsound.SND_ALIAS)