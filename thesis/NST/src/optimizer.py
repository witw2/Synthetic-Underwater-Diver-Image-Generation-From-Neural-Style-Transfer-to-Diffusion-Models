# src/optimizer.py

import torch
from project.src import train_style_transfer, load_image
from project.src.utils import preprocess_image

def optimize_parameters(content_img, style_img, num_steps, size):
    # Define possible values for each parameter
    content_weights = [5e1]
    style_weights = [1e1]
    learning_rates = [0.2, 0.5, 1.0, 2.0, 5.0]

    # Determine the device to use for computations
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the best loss as infinity
    best_loss = float('inf')
    best_params = None

    # Load and preprocess content and style images
    content_image = preprocess_image(load_image(content_img), target_size=(size, size), remove_bg=True).to(device)
    style_image = preprocess_image(load_image(style_img), target_size=(size, size)).to(device)

    # Iterate over all combinations of parameters
    for content_weight in content_weights:
        for style_weight in style_weights:
            for learning_rate in learning_rates:
                # Train the model with the current combination of parameters
                model = train_style_transfer(content_image, style_image, num_steps, content_weight, style_weight, learning_rate, optimize=True)

                # Compute the total loss
                loss = model.forward()

                # If this loss is the best so far, store this combination of parameters
                if loss < best_loss:
                    best_loss = loss
                    best_params = (content_weight, style_weight, learning_rate)

    return best_params