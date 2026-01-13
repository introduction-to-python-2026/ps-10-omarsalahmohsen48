from image_utils.py import image_load, detection_edge
from skimage.filters import median
from skimage.morphology import ball
from PIL import Image
import numpy as np

# Load image
image = image_load("your_image.jpg")  # replace with your file

# Denoise
clean_image = median(image, ball(3))

# Detect edges
edges = detection_edge(clean_image)

# Convert to binary
threshold = np.mean(edges)
binary_edges = (edges > threshold).astype(np.uint8) * 255

# Save
Image.fromarray(binary_edges).save("edge_image.png")
print("Edge image saved!")
