# main.py

import numpy as np
from skimage.filters import median
from skimage.morphology import ball
from image_utils import image_load, detection_edge
from PIL import Image

# 1. טען את תמונת הצבע
image_path = "your_image.png"  # כאן שים את מיקום התמונה שלך
image = image_load(image_path)

# 2. סנן את הרעשים בעזרת פילטר חציוני
clean_image = median(image, ball(3))

# 3. זיהוי קצוות
edge_image = detection_edge(clean_image)

# 4. המרת המערך לבינארי (0 ו-255)
threshold = np.mean(edge_image)  # אפשר גם לבחור סף אחר לפי הצורך
binary_edge = (edge_image > threshold) * 255  # 0 או 255

# 5. שמירת התמונה כ-PNG
output_image = Image.fromarray(binary_edge.astype(np.uint8))
output_image.save("edges_output.png")

# 6. הצגת התמונה
output_image.show()

print("תהליך זיהוי הקצוות הסתיים. התמונה נשמרה בשם edges_output.png")
