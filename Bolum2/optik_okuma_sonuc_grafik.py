import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cevap anahtarı örneği
cevap_anahtari = ["A", "B", "C", "D", "E", "A", "B", "C", "D", "E",
                  "A", "B", "C", "D", "E", "A", "B", "C", "D", "E"]

# Görseli oku
image = cv2.imread("images/optikForm8.png")

# Gri tonlamalı görüntüyü oluştur
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Bulanıklaştırma yap
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Eşikleme yap
thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Konturları bul
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Kabarcıkları filtreleme
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

    if max_val < 500:  # eğer işaretlenmemişse
        bos += 1
    else:
        secilen_harf = "ABCDE"[secilen_sik]
        if secilen_harf == cevap_anahtari[i]:
            dogru += 1
        else:
            yanlis += 1

# Sonuçları yazdır
plt.figure(figsize=(6, 10))
plt.bar(['Doğru', 'Yanlış', 'Boş'], [dogru, yanlis, bos], color=['green', 'red', 'gray'])
plt.title("Cevap Sonuçları")
plt.ylabel("Adet")
plt.show()

print(f"Doğru: {dogru}, Yanlış: {yanlis}, Boş: {bos}")
