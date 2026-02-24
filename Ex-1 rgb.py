# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 11:28:49 2026

@author: DELL
"""

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import time

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

    # Capture image
    if key == ord('s'):
        captured_frame = frame.copy()
        file_name = os.path.join(save_path, f"captured_{int(time.time())}.jpg")
        cv2.imwrite(file_name, captured_frame)
        print(f"Image saved at: {file_name}")
        break

    elif key == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

# =========================
# Image Enhancement Section (RGB)
# =========================
if captured_frame is not None:

    # LOG TRANSFORMATION (apply per channel)
    log_img = np.log1p(captured_frame.astype(np.float32))
    log_img = cv2.normalize(log_img, None, 0, 255, cv2.NORM_MINMAX)
    log_img = np.uint8(log_img)

    # HISTOGRAM EQUALIZATION (apply on Y channel in YCrCb)
    ycrcb = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    hist_eq = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    # GAMMA CORRECTION (apply per channel)
    gamma = 1.5
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(256)]).astype("uint8")
    gamma_img = cv2.LUT(captured_frame, table)

    # CLAHE (apply on L channel in LAB)
    lab = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    clahe_img = cv2.merge((cl, a, b))
    clahe_img = cv2.cvtColor(clahe_img, cv2.COLOR_LAB2BGR)

    # =========================
    # Display using subplot
    # =========================
    plt.figure(figsize=(12,8))

    plt.subplot(2,3,1)
    plt.imshow(cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(2,3,2)
    plt.imshow(cv2.cvtColor(log_img, cv2.COLOR_BGR2RGB))
    plt.title("Log Transformation")
    plt.axis("off")

    plt.subplot(2,3,3)
    plt.imshow(cv2.cvtColor(hist_eq, cv2.COLOR_BGR2RGB))
    plt.title("Histogram Equalization")
    plt.axis("off")

    plt.subplot(2,3,4)
    plt.imshow(cv2.cvtColor(gamma_img, cv2.COLOR_BGR2RGB))
    plt.title("Gamma Correction")
    plt.axis("off")

    plt.subplot(2,3,5)
    plt.imshow(cv2.cvtColor(clahe_img, cv2.COLOR_BGR2RGB))
    plt.title("CLAHE")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

else:
    print("No image captured.")