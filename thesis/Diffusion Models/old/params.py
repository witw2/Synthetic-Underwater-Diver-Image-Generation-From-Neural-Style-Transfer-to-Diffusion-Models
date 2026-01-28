# src/params.py

# Model Parameters
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 8
STRENGTH = 0.1
USE_CUDA = True
REMOVE_BG = True
#TEXT_CONDITIONING_PROMPT = "A high-quality underwater scene featuring a scuba diver performing clear and visible hand gestures, surrounded by realistic water textures and lighting. The water is a vibrant blue-green gradient, with beams of sunlight filtering through the surface. Fine details like bubbles, light scattering, and soft shadows create an immersive underwater atmosphere. The diver's features, suit, and equipment are highly detailed and remain clear, despite the underwater distortion. The hand gestures are precise and visible, with no blurring or loss of detail. The background features subtle underwater elements like floating particles and gentle currents, with no distractions from the diver."
#TEXT_CONDITIONING_PROMPT ="A high-quality underwater scene featuring a scuba diver performing clear and visible hand gestures, surrounded by realistic water textures and lighting. The water is a vibrant blue-green gradient, with beams of sunlight filtering through the surface. Fine details like bubbles, light scattering, and soft shadows create an immersive underwater atmosphere. The diver's features, suit, and equipment are highly detailed and remain clear, despite the underwater distortion. The hand gestures are precise and visible, with no blurring or loss of detail. The diver's right hand is positioned with the back of the hand facing the camera, showing a 'V' gesture where the index and middle fingers are extended, while the other fingers are clenched into a fist. The glove on the index finger is green, and the glove on the middle finger is white. On the back of the hand, the glove features a white square outline with an empty center. The background features subtle underwater elements like floating particles and gentle currents, with no distractions from the diver."
TEXT_CONDITIONING_PROMPT = "Apply realistic underwater textures, including soft light scattering, floating particles, and a blue-green gradient overlay, to simulate a serene underwater environment."
NEGATIVE_PROMPT = "nude, naked, skin exposure, inappropriate, explicit, erotic"

DEBUG = True

# Image Preprocessing Parameters
IMAGE_SIZE = 512

# Conditioning
USE_CONDITIONING = False
STYLE_IMAGE_PATH = "data/underwater"

# Output Paths
INPUT_DIR = "data/input"
OUTPUT_DIR = "data/output"
DEBUG_PATH = "data/debug"