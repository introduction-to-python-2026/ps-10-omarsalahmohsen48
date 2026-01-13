# image_utils.py

import numpy as np
from PIL import Image
from scipy.signal import convolve2d

def load_image(path):
    """Load an image and return as numpy array (uint8)."""
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)

def edge_detection(image):
    """Detect edges in an image (uint8 output)."""
    # 1. Convert to grayscale by averaging RGB channels
    if image.ndim == 3:
        gray = np.mean(image, axis=2).astype(np.uint8)
    else:
        gray = image

    # 2. Sobel-like filters for horizontal and vertical edges
    filterX = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    filterY = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]])

    # 3. Convolve image with filters
    edgeX = convolve2d(gray, filterX, mode='same', boundary='fill', fillvalue=0)
    edgeY = convolve2d(gray, filterY, mode='same', boundary='fill', fillvalue=0)

    # 4. Compute edge magnitude
    edge_mag = np.sqrt(edgeX**2 + edgeY**2)

    # 5. Clip and convert to uint8 (0-255)
    edge_mag = np.clip(edge_mag, 0, 255).astype(np.uint8)

    return edge_mag

