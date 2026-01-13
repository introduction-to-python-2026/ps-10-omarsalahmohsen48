from image_utils import image_load, detection_edge
from skimage.filters import median
from skimage.morphology import ball
from PIL import Image
import numpy as np

image = image_load("your_image.jpg")  # replace with your image
clean_image = median(image, ball(3))
edges = detection_edge(clean_image)
binary_edges = (edges > np.mean(edges)).astype(np.uint8) * 255
Image.fromarray(binary_edges).save("edge_image.png")
print("Edge image saved!")
