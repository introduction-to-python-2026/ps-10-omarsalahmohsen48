# main.py

import numpy as np
from skimage.filters import median
from skimage.morphology import ball
from PIL import Image
from image_utils import image_load, detection_edge

# 1. Load the color image
image_path = "your_image.png"  # replace with your image path
image = image_load(image_path)  # should return a numpy array

# 2. Apply median filter to reduce noise
clean_image = median(image, ball(3))

# 3. Detect edges
edge_image = detection_edge(clean_image)  # should return a 2D array

# 4. Convert edges to binary image
# Using mean as threshold
threshold = np.mean(edge_image)
binary_edge = (edge_image > threshold).astype(np.uint8) * 255  # 0 or 255

# 5. Save the edge image as PNG
output_image = Image.fromarray(binary_edge)
output_image.save("edges_output.png")

# 6. Display the image
output_image.show()

print("Edge detection completed. Saved as edges_o_
