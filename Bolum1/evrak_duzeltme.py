import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_document(image_path):
    image = cv2.imread(image_path)
    orig = image.copy()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 1. Gri tonlama
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Bulanıklaştırma
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Kenar algılama
    edged = cv2.Canny(blurred, 75, 200)

    # Görselleştirme: İlk adımlar
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1); plt.title("Orijinal"); plt.imshow(image_rgb); plt.axis('off')
    plt.subplot(1, 3, 2); plt.title("Gri Tonlama"); plt.imshow(gray, cmap='gray'); plt.axis('off')
    plt.subplot(1, 3, 3); plt.title("Canny Kenarlar"); plt.imshow(edged, cmap='gray'); plt.axis('off')
    plt.tight_layout(); plt.show()

    # 4. Konturlar
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    doc_cnt = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            doc_cnt = approx
            break

    if doc_cnt is None:
        print("Evrak kenarları bulunamadı.")
        return None

    # 5. Evrak kenarını çiz
    contour_image = image.copy()
    cv2.drawContours(contour_image, [doc_cnt], -1, (0, 255, 0), 2)
    plt.figure(); plt.title("Tespit Edilen Evrak Kenarı")
    plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB)); plt.axis('off'); plt.show()

    # 6. Perspektif düzeltme
    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def four_point_transform(image, pts):
        rect = order_points(pts.reshape(4, 2))
        (tl, tr, br, bl) = rect
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxWidth = max(int(widthA), int(widthB))
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([[0, 0], [maxWidth - 1, 0], 
                        [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped

    warped = four_point_transform(orig, doc_cnt)
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)

    # Görselleştirme - Perspektif düzeltme
    plt.figure(); plt.title("Perspektif Düzeltilmiş")
    plt.imshow(warped_rgb); plt.axis('off'); plt.show()

    # 7. Kontrast Artırma - CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    clahe_applied = clahe.apply(warped_gray)

    # 8. Otsu ile eşikleme
    _, thresh_clahe = cv2.threshold(clahe_applied, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 9. Gürültü Temizleme
    kernel = np.ones((2,2), np.uint8)
    cleaned = cv2.morphologyEx(thresh_clahe, cv2.MORPH_OPEN, kernel)

    # Görselleştirme - Karşılaştırmalı
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.title("CLAHE Sonrası"); plt.imshow(clahe_applied, cmap='gray'); plt.axis('off')
    plt.subplot(1, 3, 2); plt.title("CLAHE + Otsu"); plt.imshow(thresh_clahe, cmap='gray'); plt.axis('off')
    plt.subplot(1, 3, 3); plt.title("Gürültü Temizlenmiş"); plt.imshow(cleaned, cmap='gray'); plt.axis('off')
    plt.tight_layout(); plt.show()

    return cleaned  # Nihai ikili görüntü döndürülür

# Örnek kullanım
preprocess_document("images/evrak4.jpg")
