# src/params.py

# Model Parameters
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
USE_CUDA = True

DEBUG = False

# Image Processing Parameters
IMAGE_SIZE = (512, 544)
COLOR_MIN = (0, 0, 169)
COLOR_MAX = (31, 149, 169)
SCALE_FACTOR = 0.7
SCORE=0.8
INTRODUCE_VARIATION = True

# Output Paths
INPUT_DIR = "data/input"
OUTPUT_DIR = "data/output"
DEBUG_PATH = "data/debug"
BASE_PATH = "data/underwater"
