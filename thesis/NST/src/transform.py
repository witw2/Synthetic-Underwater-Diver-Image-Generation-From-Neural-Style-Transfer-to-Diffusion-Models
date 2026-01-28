# src/transform.py

import torch
from .model import train_style_transfer
from .data_manager import load_image, save_image
from .utils import preprocess_image, postprocess_image, read_dict_from_file

# Read parameters from a file
params=read_dict_from_file()

def transform_image(content_image_path, style_image_pil, output_image_path, size=params['size']):
    # Determine the device to use for computations
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load and preprocess content image
    content_img = preprocess_image(load_image(content_image_path), target_size=size, remove_bg=True,should_save=params["debug"]).to(device)

    # Convert PIL style image to tensor
    style_img = preprocess_image(style_image_pil, target_size=size).to(device)

    # Perform style transfer
    with torch.no_grad():
        output_img = train_style_transfer(content_img, style_img,debug=params["debug"],plot=params["plot"]).to(device)

    # Post-process and save the image
    output_img = postprocess_image(output_img.cpu())
    if params["debug"]:
        output_img.show()
    save_image(output_img, output_image_path)