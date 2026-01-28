# src/model.py
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import matplotlib.pyplot as plt
from .data_manager import save_image
from .utils import postprocess_image, read_dict_from_file

# Read parameters from a file
params=read_dict_from_file()

# Set weights for content, style and total variation
c_w=params['content_weight']
s_w=params['style_weight']
t_v_w=params['total_variation_weight']

# Define a class for VGG19 feature extraction
class VGG19FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG19FeatureExtractor, self).__init__()
        # Load the pretrained VGG19 model
        vgg19 = models.vgg19(pretrained=True).features
        # Define the layers to extract features from
        self.layers = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2',  # Content layer
            '28': 'conv5_1'
        }
        # Create a sequential model with the selected layers
        self.model = nn.Sequential(*list(vgg19.children())[:29])

    def forward(self, x):
        # Extract features from the input tensor
        features = {}
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in self.layers:
                features[self.layers[name]] = x
        return features

# Define a class for the neural style transfer model
class NeuralStyleTransferModel(nn.Module):
    def __init__(self, content_img, style_img, content_weight=c_w, style_weight=s_w, total_variation_weight=t_v_w):
        super(NeuralStyleTransferModel, self).__init__()
        # Set the weights for content, style and total variation
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.total_variation_weight = total_variation_weight
        # Initialize the feature extractor
        self.feature_extractor = VGG19FeatureExtractor().eval().to(content_img.device)
        # Initialize the target image as a copy of the content image
        self.target = nn.Parameter(content_img.clone(), requires_grad=True).to(content_img.device)
        # Set the content and style images
        self.content_img = content_img
        self.style_img = style_img

        # Extract features from the style and content images
        self.style_features = self.feature_extractor(style_img)
        self.content_features = self.feature_extractor(content_img)

        # Compute the gram matrices for the style features
        self.style_grams = {layer: self.gram_matrix(self.style_features[layer]) for layer in self.style_features}

    def gram_matrix(self, input):
        # Compute the gram matrix for a given input tensor
        _, c, h, w = input.size()
        features = input.view(c, h * w)
        G = torch.mm(features, features.t())
        return G.div(c * h * w)

    def compute_content_loss(self, target_features):
        # Compute the content loss between the target and content features
        content_loss = torch.mean((target_features['conv4_2'] - self.content_features['conv4_2']) ** 2)
        return content_loss

    def compute_style_loss(self, target_features):
        # Compute the style loss between the target and style features
        style_loss = 0
        for layer in self.style_grams:
            target_feature = target_features[layer]
            target_gram = self.gram_matrix(target_feature)
            style_gram = self.style_grams[layer]
            style_loss += torch.mean((target_gram - style_gram) ** 2)
        return style_loss

    def compute_variation_loss(self):
        # Compute the total variation loss for the target image
        diff_i = torch.sum(torch.abs(self.target[:, :, :, 1:] - self.target[:, :, :, :-1]))
        diff_j = torch.sum(torch.abs(self.target[:, :, 1:, :] - self.target[:, :, :-1, :]))
        return diff_i + diff_j

    def forward(self):
        # Compute the total loss for the neural style transfer model
        target_features = self.feature_extractor(self.target)
        content_loss = self.compute_content_loss(target_features)
        style_loss = self.compute_style_loss(target_features)
        total_variation_loss = self.compute_variation_loss()
        total_loss = self.content_weight * content_loss + self.style_weight * style_loss + self.total_variation_weight * total_variation_loss
        return total_loss

# Define a function to train the neural style transfer model
def train_style_transfer(content_img, style_img, num_steps=params["num_steps"], content_weight=c_w, style_weight=s_w, total_variation_weight=t_v_w, learning_rate=params["learning_rate"],debug=False,plot=False,optimize=False):
    # Initialize the neural style transfer model
    model = NeuralStyleTransferModel(content_img, style_img, content_weight, style_weight, total_variation_weight)
    model.to(content_img.device)
    # Initialize the optimizer
    optimizer = optim.Adam([model.target], lr=learning_rate)
    torch.set_grad_enabled(True)

    # Initialize lists to store the losses
    total_losses = []
    content_losses = []
    style_losses = []

    # Start the timer
    start_time = time.time()

    # Train the model
    for step in range(num_steps):
        # Zero the gradients
        optimizer.zero_grad()
        # Compute the lossz
        loss = model()
        # Backpropagate the loss
        loss.backward(retain_graph=True)
        # Update the weights
        optimizer.step()

        # Print the losses every 50 steps
        if step % 50 == 0:
            elapsed_time = time.time() - start_time
            avg_time_per_step = elapsed_time / (step + 1)
            estimated_time_remaining = avg_time_per_step * (num_steps - step)

            # Compute the content, style and total variation losses
            content_loss = model.compute_content_loss(model.feature_extractor(model.target))
            style_loss = model.compute_style_loss(model.feature_extractor(model.target))
            total_variation_loss= model.compute_variation_loss()
            if debug:
                print(f"\rStep: {step}/{num_steps}\t"
                      f"Total Loss: {loss.item():.2f}\t"
                      f"Content Loss: {content_loss.item():.2f}\t"
                      f"Style Loss: {style_loss.item():.2f}\t"
                    f"Total Variation Loss: {total_variation_loss.item():.2f}\t"
                      f"Estimated Time Remaining: {estimated_time_remaining:.2f} seconds\t",end='')

            # Store the losses
            total_losses.append(loss.item())
            content_losses.append(content_loss.item())
            style_losses.append(style_loss.item())
    if debug:
        print()
    if plot:
        # Plot the losses
        plt.figure(figsize=(10, 5))
        plt.plot(total_losses, label='Total Loss')
        plt.plot(content_losses, label='Content Loss')
        plt.plot(style_losses, label='Style Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

    if debug:
        # Save the final image
        script_dir = os.path.dirname(os.path.realpath(__file__))
        script_dir=script_dir[:-4]
        opath = os.path.join(script_dir, f'data/output/subOut/{num_steps}steps{content_weight}c_w{style_weight}s_w{learning_rate}lr{loss.item():.2f}total{content_loss.item():.2f}conent{style_loss.item():.2f}style{params["style"]}NOstyle.jpg')
        save_image(postprocess_image(model.target.cpu()), opath)
        #make .txt file with loss values
        with open(os.path.join(script_dir, 'data.txt'), 'w') as f:
            f.write(f"Total Loss: {loss.item():.2f}\n"
                    f"Content Loss: {content_loss.item():.2f}\n"
                    f"Style Loss: {style_loss.item():.2f}\n"
                    f"Total Variation Loss: {total_variation_loss.item():.2f}\n"
                    f"Time Elapsed: {elapsed_time:.2f}s\n"
                    f"Learning Rate: {learning_rate}\n"
                    f"Content Weight: {content_weight}\n"
                    f"Style Weight: {style_weight}\n"
                    f"Total Variation Weight: {total_variation_weight}\n"
                    f"Number of Steps: {num_steps}\n"
                    f"Style: {params['style']}\n")

    if optimize:
        return model

    return model.target