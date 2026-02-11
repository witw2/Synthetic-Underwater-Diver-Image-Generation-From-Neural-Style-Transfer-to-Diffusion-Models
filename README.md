# Synthetic Underwater Diver Image Generation

## Purpose
This project aims to generate synthetic underwater diver images using advanced machine learning techniques such as Neural Style Transfer and Diffusion Models. The goal is to improve the dataset available for training algorithms in underwater imaging and enhance the realism of generated images.

## Methodology

### Neural Style Transfer with VGG19
Neural Style Transfer (NST) allows the reimagining of images by applying the style of one image to the content of another. In this project, we leverage the VGG19 convolutional neural network, pre-trained on ImageNet, to extract features from both content and style images. The steps include:
1. **Content Representation**: Extracting features from input diver images.
2. **Style Representation**: Extracting texture and color characteristics from style images.
3. **Image Reconstruction**: Combining both features to generate styled underwater diver images.

### Diffusion Models
Diffusion models are generative models that learn to generate new data by reversing a diffusion process. These models gradually add noise to the data until it becomes indistinguishable from pure noise, and then learn to reconstruct the original data. The specific steps are:
1. **Noise Addition**: Gradually corrupting training images by adding Gaussian noise.
2. **Denoising Process**: Training a model to reverse the corruption process and regenerate realistic images.
3. **Sampling**: Using the trained model to produce high-quality synthetic images.

## Features
- Generation of realistic underwater diver images.
- Utilization of advanced techniques: Neural Style Transfer and Diffusion Models.
- Flexibility to apply different artistic styles through NST.
- Model training and testing scripts included.

## Installation
To set up the project, follow these steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/witw2/Synthetic-Underwater-Diver-Image-Generation-From-Neural-Style-Transfer-to-Diffusion-Models.git
   cd Synthetic-Underwater-Diver-Image-Generation-From-Neural-Style-Transfer-to-Diffusion-Models
