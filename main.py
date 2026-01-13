# main.py
import numpy as np
from PIL import Image
from skimage.filters import median
from skimage.morphology import ball
from image_utils import load_image, edge_detection

# --- Configuration ---
IMAGE_PATH = '.tests/lena.jpg'           # Updated input image path
OUTPUT_IMAGE_NAME = 'my_edges.png'       # Output filename for the edge image
MEDIAN_FILTER_SIZE = 3                    # Median filter size for noise suppression
EDGE_DETECTION_THRESHOLD = 50             # Threshold for binary edge image

# --- Step 1: Load the original image ---
original_image = load_image(IMAGE_PATH)

if original_image is None:
    raise FileNotFoundError(f"Failed to load image from {IMAGE_PATH}")

# --- Step 2: Noise suppression using a median filter ---
clean_image = median(original_image, ball(MEDIAN_FILTER_SIZE))

# --- Step 3: Detect edges ---
edge_magnitude = edge_detection(clean_image)

# --- Step 4: Convert edge magnitude to binary image ---
edge_binary = (edge_magnitude > EDGE_DETECTION_THRESHOLD).astype(np.uint8) * 255

# --- Step 5: Save the resulting binary edge image ---
edge_image_pil = Image.fromarray(edge_binary)
edge_image_pil.save(OUTPUT_IMAGE_NAME)

print(f"Edge-detected image saved as '{OUTPUT_IMAGE_NAME}'")
