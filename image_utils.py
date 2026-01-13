from PIL import Image
import numpy as np
from scipy.signal import convolve2d

def load_image(path):
    img = Image.open(path).convert('L')  
    return np.array(img)

def edge_detection(image):
    
    Kx = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]])
    Ky = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])
    

    gx = convolve2d(image, Kx, mode='same', boundary='symm')
    gy = convolve2d(image, Ky, mode='same', boundary='symm')
    

    edges = np.sqrt(gx**2 + gy**2)

    edges = (edges / edges.max()) * 255
    return edges.astype(np.uint8)
