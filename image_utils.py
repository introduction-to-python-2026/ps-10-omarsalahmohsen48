import numpy as np
from scipy.signal import convolve2d
from PIL import Image

def image_load(image_path):
    """
    Load a color image and convert it to a NumPy array.
    """
    image = Image.open(image_path).convert("RGB")
    return np.array(image)


def detection_edge(image_array):
    """
    Detect edges in an image array.
    """
    # Convert to grayscale (max value across RGB channels)
    gray = np.max(image_array, axis=2)

    # Horizontal and vertical filters (Sobel)
    filter_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])
    filter_y = np.array([[-1, -2, -1],
                         [0, 0, 0],
                         [1, 2, 1]])
    
    # Convolve filters with image
    edge_x = convolve2d(gray, filter_x, mode='same', boundary='fill', fillvalue=0)
    edge_y = convolve2d(gray, filter_y, mode='same', boundary='fill', fillvalue=0)
    
    # Compute edge magnitude
    edge_mag = np.sqrt(edge_x**2 + edge_y**2)
    
    return edge_mag
