# -*- coding: utf-8 -*-

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Folder to save image
save_path = "captured_images"
os.makedirs(save_path, exist_ok=True)

# Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not access the webcam")
    exit()

print("Press 's' to capture and process image")
print("Press 'q' to quit")

captured_frame = None

while True:
    ret, frame = camera.read()
    if not ret:
        print("Failed to capture image")
        break

    cv2.imshow("Webcam Preview", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        captured_frame = frame.copy()
        file_name = os.path.join(save_path, "captured_image.jpg")
        cv2.imwrite(file_name, captured_frame)
        print(f"Image saved at: {file_name}")
        break

    elif key == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

# =========================
# Image Enhancement Section
# =========================
if captured_frame is not None:

    # Convert to grayscale
    gray = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2GRAY)

    # LOG TRANSFORMATION
    log_img = np.log1p(gray.astype(np.float32))
    log_img = cv2.normalize(log_img, None, 0, 255, cv2.NORM_MINMAX)
    log_img = np.uint8(log_img)

    # HISTOGRAM EQUALIZATION
    hist_eq = cv2.equalizeHist(gray)

    # GAMMA CORRECTION
    gamma = 1.5
    gamma_img = np.array(255 * (gray / 255) ** gamma, dtype='uint8')

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)

    # Convert original to RGB for display
    original_rgb = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB)

    # Prepare images list
    images = [
        ("Original", cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)),
        ("Grayscale", gray),
        ("Log Transform", log_img),
        ("Histogram Equalization", hist_eq),
        ("Gamma Correction", gamma_img),
        ("CLAHE", clahe_img)
    ]

    # =========================
    # Display Images + Histograms
    # =========================
    plt.figure(figsize=(14, 10))

    for i, (title, img) in enumerate(images):
        # Image subplot
        plt.subplot(len(images), 2, 2*i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis("off")

        # Histogram subplot
        plt.subplot(len(images), 2, 2*i + 2)
        plt.hist(img.ravel(), bins=256)
        plt.title(f"{title} Histogram")

    plt.tight_layout()
    plt.show()

else:
    print("No image captured.")