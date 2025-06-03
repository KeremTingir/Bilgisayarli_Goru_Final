import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Görüntüyü yükleme
image = cv2.imread('images/mercimek.jpg')
original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 2. Gri tonlamaya çevirme
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 3. Gaussian bulanıklaştırma
blurred = cv2.GaussianBlur(gray, (11, 11), 0)

# 4. Eşikleme
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 5. Morfolojik açma (gürültü temizleme)
kernel = np.ones((3,3), np.uint8)
opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# 6. Kontur bulma
contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
count = len(contours)

# Konturları çizme
output = original.copy()
cv2.drawContours(output, contours, -1, (0, 255, 0), 2)

# 7. Görselleştirme
titles = ['Orijinal Görüntü', 'Gri Tonlama', 'Gaussian Blur', 'Eşikleme', 'Açma (Morfolojik)', f'Sonuç - Sayı: {count}']
images = [original, gray, blurred, thresh, opened, output]

plt.figure(figsize=(15, 10))
for i in range(6):
    plt.subplot(2, 3, i+1)
    if i == 1 or i == 2 or i == 3 or i == 4:
        plt.imshow(images[i], cmap='gray')
    else:
        plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()

print(f"Toplam mercimek tanesi sayısı: {count}")
