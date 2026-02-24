import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1️⃣ Read Image
img = cv2.imread(r'C:\Users\STUDENT\Downloads\Fig0429(a)(blown_ic).tif', 0)
print(img is None)


# 2️⃣ FFT - Convert to Frequency Domain
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# Magnitude Spectrum (Output 1)
magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

rows, cols = img.shape
crow, ccol = rows // 2, cols // 2

# 3️⃣ Create LPF and HPF Masks
mask_lpf = np.zeros((rows, cols), np.uint8)
mask_hpf = np.ones((rows, cols), np.uint8)

r = 30  # radius

for i in range(rows):
    for j in range(cols):
        if (i - crow)**2 + (j - ccol)**2 <= r*r:
            mask_lpf[i, j] = 1
            mask_hpf[i, j] = 0

# 4️⃣ Apply Masks (Output 2)
fshift_lpf = fshift * mask_lpf
fshift_hpf = fshift * mask_hpf

magnitude_lpf = 20 * np.log(np.abs(fshift_lpf) + 1)
magnitude_hpf = 20 * np.log(np.abs(fshift_hpf) + 1)

# 5️⃣ Inverse FFT (Output 3)
img_lpf = np.fft.ifft2(np.fft.ifftshift(fshift_lpf))
img_lpf = np.abs(img_lpf)

img_hpf = np.fft.ifft2(np.fft.ifftshift(fshift_hpf))
img_hpf = np.abs(img_hpf)

# 6️⃣ Display All Outputs
plt.figure(figsize=(12,10))

plt.subplot(3,2,1), plt.imshow(img, cmap='gray'), plt.title('Original Image')
plt.subplot(3,2,2), plt.imshow(magnitude_spectrum, cmap='gray'), plt.title('FFT Spectrum')

plt.subplot(3,2,3), plt.imshow(magnitude_lpf, cmap='gray'), plt.title('LPF in Frequency Domain')
plt.subplot(3,2,4), plt.imshow(magnitude_hpf, cmap='gray'), plt.title('HPF in Frequency Domain')

plt.subplot(3,2,5), plt.imshow(img_lpf, cmap='gray'), plt.title('LPF - Spatial Domain')
plt.subplot(3,2,6), plt.imshow(img_hpf, cmap='gray'), plt.title('HPF - Spatial Domain')

plt.tight_layout()
plt.show()
