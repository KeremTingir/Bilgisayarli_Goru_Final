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

    # Görselleştirme: Orijinal ve Kenar Algılama
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1); plt.title("Orijinal Görüntü"); plt.imshow(image_rgb); plt.axis('off')
    plt.subplot(1, 3, 2); plt.title("Gri Tonlama"); plt.imshow(gray, cmap='gray'); plt.axis('off')
    plt.subplot(1, 3, 3); plt.title("Kenar Algılama (Canny)"); plt.imshow(edged, cmap='gray'); plt.axis('off')
    plt.tight_layout(); plt.show()

    # Kontur bulma
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

    # Kenarı çiz
    contour_image = image.copy()
    cv2.drawContours(contour_image, [doc_cnt], -1, (0, 255, 0), 2)
    plt.figure(); plt.title("Tespit Edilen Evrak Kenarı")
    plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB)); plt.axis('off'); plt.show()

    # Perspektif düzeltme
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

    # Görüntüyü dönüştür
    warped = four_point_transform(orig, doc_cnt)
    warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    # Kontrast artırma yöntemleri
    # Yöntem 1: Basit sabit eşik
    _, bin_thresh = cv2.threshold(warped_gray, 180, 255, cv2.THRESH_BINARY)

    # Yöntem 2: Otsu threshold
    _, otsu_thresh = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Yöntem 3: Gamma düzeltme + Otsu
    warped_float = warped_gray.astype(np.float32) / 255.0
    warped_gamma = cv2.pow(warped_float, 0.8)
    gamma_corrected = (warped_gamma * 255).astype(np.uint8)
    _, gamma_thresh = cv2.threshold(gamma_corrected, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Görselleştirme - Tüm yöntemler karşılaştırmalı
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 4, 1); plt.title("Perspektif Düzeltilmiş"); plt.imshow(warped_rgb); plt.axis('off')
    plt.subplot(1, 4, 2); plt.title("Basit Threshold"); plt.imshow(bin_thresh, cmap='gray'); plt.axis('off')
    plt.subplot(1, 4, 3); plt.title("Otsu Threshold"); plt.imshow(otsu_thresh, cmap='gray'); plt.axis('off')
    plt.subplot(1, 4, 4); plt.title("Gamma + Otsu"); plt.imshow(gamma_thresh, cmap='gray'); plt.axis('off')
    plt.tight_layout(); plt.show()

    return gamma_thresh  # İstersen diğerlerinden biri de döndürülebilir

# Örnek kullanım
preprocess_document("images/evrak.jpg")
