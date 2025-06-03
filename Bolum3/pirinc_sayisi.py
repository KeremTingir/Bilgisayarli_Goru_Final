import cv2
import numpy as np
import matplotlib.pyplot as plt

# Görseli oku
image = cv2.imread('images/pirinc.jpg')
original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Renk kanallarını ayır
b, g, r = cv2.split(image)

# Kanal seçimi (örneğin mavi kanalda pirinçler daha belirgin olabilir)
channel = b  # veya b ya da r, en iyi sonucu deneyerek görebiliriz

# Kanal üzerinde eşikleme (ters binary çünkü pirinçler açık renk)
_, thresh = cv2.threshold(channel, 180, 255, cv2.THRESH_BINARY_INV)

# Açma işlemi ile küçük gürültüleri temizle
kernel = np.ones((3,3), np.uint8)
opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# Kontur bul
contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Alan filtresi ile sahte nesneleri çıkar
min_area = 400
max_area = 1500
filtered_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]

# Çizim
output = original.copy()
cv2.drawContours(output, filtered_contours, -1, (0, 255, 0), 2)

# Sayım
count = len(filtered_contours)

# Görselleştirme
titles = ['Orijinal', 'Seçilen Kanal', 'Eşikleme', f'Sonuç - Sayı: {count}']
images = [original, channel, thresh, output]

plt.figure(figsize=(16, 6))
for i in range(4):
    plt.subplot(1, 4, i+1)
    cmap = 'gray' if len(images[i].shape) == 2 else None
    plt.imshow(images[i], cmap=cmap)
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()

print(f"Pirinç tanesi sayısı (filtreli): {count}")
