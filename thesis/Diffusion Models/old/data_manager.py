# src/utils.py
import os
from params import REMOVE_BG, OUTPUT_DIR, DEBUG_PATH, DEBUG, NUM_DIFFUSION_STEPS, GUIDANCE_SCALE, STRENGTH, TEXT_CONDITIONING_PROMPT
import os
from PIL import PngImagePlugin

'''
def preprocess_image(image):
    """Resize and normalize the image for the model, ensuring it returns a PIL Image."""
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))  # Resize to the desired input size
    return image  # Return as a PIL Image
'''




def save_image(image, filename, num_steps=NUM_DIFFUSION_STEPS, guidance_scale=GUIDANCE_SCALE, strength=STRENGTH, prompt=TEXT_CONDITIONING_PROMPT):
    """Save the image to the output directory and optionally save debug information."""
    # Ensure the output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Save the image to the primary output directory
    image.save(os.path.join(OUTPUT_DIR, filename))

    # If DEBUG is enabled, save to DEBUG_PATH with detailed metadata
    if DEBUG:
        if not os.path.exists(DEBUG_PATH):
            os.makedirs(DEBUG_PATH)

        # Create debug filename including num_steps, guidance_scale, and strength
        debug_filename = (
            f"{filename.split('.')[0]}_steps{num_steps}_scale{guidance_scale}_strength{strength}.png"
        )
        debug_filepath = os.path.join(DEBUG_PATH, debug_filename)

        # Add metadata
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("Prompt", prompt or "No prompt provided")
        metadata.add_text("Num Steps", str(num_steps or "N/A"))
        metadata.add_text("Guidance Scale", str(guidance_scale or "N/A"))
        metadata.add_text("Strength", str(strength or "N/A"))
        metadata.add_text("Remove BG", str(REMOVE_BG or "N/A"))

        # Save image with metadata
        image.save(debug_filepath, "PNG", pnginfo=metadata)

