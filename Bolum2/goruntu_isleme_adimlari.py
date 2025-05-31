import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cevap anahtarı örneği
cevap_anahtari = ["A", "B", "C", "D", "E", "A", "B", "C", "D", "E",
                  "A", "B", "C", "D", "E", "A", "B", "C", "D", "E"]

# Görseli oku
image = cv2.imread("images/optikForm8.png")
if image is None:
    raise FileNotFoundError("Görüntü dosyası 'images/optikForm2.png' bulunamadı.")

# Adım 1: Orijinal görüntüyü görselleştir
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(8, 6))
plt.imshow(image_rgb)
plt.title("Adım 1: Orijinal Görüntü")
plt.axis('off')
plt.show()

# Adım 2: Gri tonlamalı görüntüyü oluştur ve görselleştir
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.figure(figsize=(8, 6))
plt.imshow(gray, cmap='gray')
plt.title("Adım 2: Gri Tonlamalı")
plt.axis('off')
plt.show()

# Adım 3: Bulanıklaştırma yap ve görselleştir
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
plt.figure(figsize=(8, 6))
plt.imshow(blurred, cmap='gray')
plt.title("Adım 3: Bulanıklaştırılmış")
plt.axis('off')
plt.show()

# Adım 4: Eşikleme yap ve görselleştir
thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
plt.figure(figsize=(8, 6))
plt.imshow(thresh, cmap='gray')
plt.title("Adım 4: Eşiklenmiş (Threshold)")
plt.axis('off')
plt.show()

# Konturları bul
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

bubble_contours = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    ar = w / float(h)
    if 20 <= w <= 50 and 20 <= h <= 50 and 0.9 <= ar <= 1.1:
        bubble_contours.append(cnt)

# Kabarcıkları sıralama: yukarıdan aşağıya, sonra soldan sağa
def sort_contours(cnts, rows=20, cols=5):
    bounding_boxes = [cv2.boundingRect(c) for c in cnts]
    cnts = [x for _, x in sorted(zip(bounding_boxes, cnts), key=lambda b: b[0][1])]
    grouped = [cnts[i * cols:(i + 1) * cols] for i in range(rows)]
    for group in grouped:
        group.sort(key=lambda c: cv2.boundingRect(c)[0])
    return grouped

sorted_bubbles = sort_contours(bubble_contours)

# Sonuç görseli için orijinal görüntünün kopyasını oluştur
result_image = image.copy()

# Cevapları algıla ve karşılaştır
dogru = yanlis = bos = 0
for i, row in enumerate(sorted_bubbles):
    max_val = 0
    secilen_sik = -1
    for j, c in enumerate(row):
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        total = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
        if total > max_val:
            max_val = total
            secilen_sik = j

        # Konturları sonuç görseline çiz
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 1)

    if max_val < 500:  # eğer işaretlenmemişse
        bos += 1
    else:
        secilen_harf = "ABCDE"[secilen_sik]
        if secilen_harf == cevap_anahtari[i]:
            dogru += 1
            # Doğru cevapları yeşil ile işaretle
            x, y, w, h = cv2.boundingRect(row[secilen_sik])
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            yanlis += 1
            # Yanlış cevapları kırmızı ile işaretle
            x, y, w, h = cv2.boundingRect(row[secilen_sik])
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Adım 5: Sonuç görüntüsünü görselleştir
result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(8, 6))
plt.imshow(result_image_rgb)
plt.title("Adım 5: Sonuç (Doğru: Yeşil, Yanlış: Kırmızı)")
plt.axis('off')
plt.show()

# Adım 6: Sonuç metnini görselleştir
plt.figure(figsize=(8, 6))
plt.text(0.5, 0.5, f"Doğru: {dogru}\nYanlış: {yanlis}\nBoş: {bos}",
         fontsize=12, ha='center', va='center')
plt.title("Adım 6: Sonuçlar")
plt.axis('off')
plt.show()

# Konsola sonuçları yazdır
print(f"Doğru: {dogru}, Yanlış: {yanlis}, Boş: {bos}")