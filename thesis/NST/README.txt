# Neural Style Transfer - diver photos

This project is a Python-based implementation of the Neural Style Transfer algorithm. It uses a pre-trained VGG19 model to extract features from content and style images, and then optimizes a target image to match the content of the content image and the style of the style image.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

The project requires the following Python libraries:

- torch
- torchvision
- PIL
- numpy
- winsound
- rembg
- cv2

You can install these libraries using pip:

```plaintext
pip install torch torchvision pillow numpy winsound rembg opencv-python-headless
```

### Configuration

The project uses a configuration file named `params.txt` to set various parameters for the style transfer process. The file is in JSON format and contains the following keys:

- `content_weight`: The weight of the content loss in the total loss computation.
- `style_weight`: The weight of the style loss in the total loss computation.
- `learning_rate`: The learning rate for the optimizer.
- `num_steps`: The number of steps to run the optimizer.
- `size`: The size to resize the input images to, specified as a list of two integers [height, width].
- `style`: The style to use for the style transfer, specified as an subdirectory name in style directory.
- `total_variation_weight`: The weight of the total variation loss in the total loss computation.
- `debug`: A boolean flag that, when set to true, enables debug mode which may print additional information during the style transfer process.

You can modify these parameters to customize the behavior of the style transfer process.

### Usage

The main entry point of the application is the `main()` function in the `main.py` file. This function reads the content and style images from the specified directories, combines the style images into one, and then applies the style transfer to each content image. The transformed images are saved in the output directory.

You can run the application with the following command:

```plaintext
python main.py
```

## Acknowledgments

- The project uses a pre-trained VGG19 model from the torchvision library.
- The project uses the rembg library to remove the background from images.
- The project uses the OpenCV library for image processing tasks.