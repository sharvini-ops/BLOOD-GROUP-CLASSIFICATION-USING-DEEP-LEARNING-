import os
import cv2
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Save inside dataset
save_folder = "dataset/processed"

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

Tk().withdraw()

file_path = askopenfilename(
    title="Select Image",
    filetypes=[("Image Files","*.jpg *.jpeg *.png *.bmp")]
)

if not file_path:
    print("No file selected")
    exit()

img = cv2.imread(file_path)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ✅ IMPORTANT FIX
resized = cv2.resize(gray, (224,224))

# Enhance
enhanced = cv2.equalizeHist(resized)

filename = os.path.basename(file_path)
save_path = os.path.join(save_folder, "processed_" + filename)

cv2.imwrite(save_path, enhanced)

print("\nImage saved at:", save_path)

cv2.imshow("Processed Image", enhanced)
cv2.waitKey(0)
cv2.destroyAllWindows()