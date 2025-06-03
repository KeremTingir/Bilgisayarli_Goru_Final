import cv2
import numpy as np
import matplotlib.pyplot as plt

# Görüntüyü yükle
image = cv2.imread('images/nohut.jpg')
original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Griye çevir + Gaussian blur
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (11, 11), 0)

# Eşikleme
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Morfolojik açma
kernel = np.ones((3,3), np.uint8)
opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# Konturları bul
contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Alan filtresi: sadece belirli büyüklükteki konturlar
min_area = 1000  # küçük gürültüyü elemek için
max_area = 6000  # aşırı büyük şeyleri dışlamak için

filtered_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]

# Sayım
count = len(filtered_contours)

# Konturları çiz
output = original.copy()
cv2.drawContours(output, filtered_contours, -1, (0, 255, 0), 2)

# Görselleştirme
titles = ['Orijinal Görüntü', 'Eşikleme', 'Açma', f'Filtreli Konturlar - Sayı: {count}']
images = [original, thresh, opened, output]

plt.figure(figsize=(15, 8))
for i in range(4):
    plt.subplot(1, 4, i+1)
    cmap = 'gray' if len(images[i].shape) == 2 else None
    plt.imshow(images[i], cmap=cmap)
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()

print(f"Nohut sayısı (filtreli): {count}")
