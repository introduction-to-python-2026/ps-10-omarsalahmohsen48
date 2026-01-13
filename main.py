# main.py

import numpy as np
from skimage.filters import median
from skimage.morphology import ball
from PIL import Image
from image_utils import image_load, detection_edge

# 1. טען את תמונת הצבע
image_path = "your_image.png"  # תעדכן למיקום הקובץ שלך
image = image_load(image_path)

# 2. סנן את הרעשים בעזרת פילטר חציוני
clean_image = median(image, ball(3))

# 3. הרץ זיהוי קצוות
edge_image = detection_edge(clean_image)

# 4. המרת מערך הקצוות לבינארי לפי סף
threshold = np.percentile(edge_image, 75)  # סף דינמי לפי ערכי התמונה
binary_edge = (edge_image > threshold).astype(np.uint8) * 255  # 0 או 255

# 5. שמירת התמונה כ-PNG
output_image = Image.fromarray(binary_edge)
output_image.save("edges_output.png")

# 6. הצגת התמונה
output_image.show()

print("תהליך זיהוי הקצוות הסתיים. התמונה נשמרה בשם edges_output.png")
