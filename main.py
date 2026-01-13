# main.py

from skimage.filters import median
from skimage.morphology import ball
from PIL import Image
from image_utils import load_image, edge_detection

# Load image
image = load_image("your_image.png")

# Remove noise
clean_image = median(image, ball(3))

# Detect edges
edge_image = edge_detection(clean_image)

# Save edge image
Image.fromarray(edge_image).save("edges_output.png")
Image.fromarray(edge_image).show()
