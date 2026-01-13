from image_utils import load_image, detection_edge
from skimage.filters import median
from skimage.morphology import ball
import numpy as np
from PIL import Image

# Step 1: Load the original image
image_path = "your_image.jpg"  # <-- replace with your image file
image = load_image(image_path)

# Step 2: Denoise the image using median filter
clean_image = median(image, ball(3))  # you can adjust the ball size

# Step 3: Detect edges
edge_image = detection_edge(clean_image)

# Step 4: Convert edge image to binary using threshold
threshold = np.mean(edge_image)  # simple threshold
binary_edge = edge_image > threshold  # True / False array

# Step 5: Save the resulting edge image
binary_edge_uint8 = (binary_edge * 255).astype(np.uint8)
edge_pil = Image.fromarray(binary_edge_uint8)
edge_pil.save("edge_image.png")

# Optional: Display images
edge_pil.show()
