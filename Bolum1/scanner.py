import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_document(image_path):
    # 1. Görüntüyü Yükleme
    image = cv2.imread(image_path)
    if image is None:
        print("Görüntü yüklenemedi.")
        return None
    orig = image.copy()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 2. Gri Tonlama
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 3. Bulanıklaştırma
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 4. Kenar Algılama (Canny)
    edged = cv2.Canny(blurred, 75, 200)

    # Görselleştirme: İlk Adımlar
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1); plt.title("Orijinal Görüntü"); plt.imshow(image_rgb); plt.axis('off')
    plt.subplot(1, 3, 2); plt.title("Gri Tonlama"); plt.imshow(gray, cmap='gray'); plt.axis('off')
    plt.subplot(1, 3, 3); plt.title("Canny Kenarlar"); plt.imshow(edged, cmap='gray'); plt.axis('off')
    plt.tight_layout(); plt.savefig('results/Bolum1/step1_initial_processing.png'); plt.close()

    # 5. Kontur Bulma
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

    # 6. Evrak Kenarını Çizme
    contour_image = image.copy()
    cv2.drawContours(contour_image, [doc_cnt], -1, (0, 255, 0), 2)

    # Görselleştirme: Kontur Tespiti
    plt.figure(); plt.title("Tespit Edilen Evrak Kenarı")
    plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB)); plt.axis('off')
    plt.savefig('results/Bolum1/step2_contour_detection.png'); plt.close()

    # 7. Perspektif Düzeltme
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
    warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    # Görselleştirme: Perspektif Düzeltme
    plt.figure(); plt.title("Perspektif Düzeltilmiş")
    plt.imshow(warped_rgb); plt.axis('off')
    plt.savefig('results/Bolum1/step3_perspective_corrected.png'); plt.close()

    # 8. Kontrast Artırma Yöntemleri
    # Yöntem 1: Basit Sabit Eşik
    _, bin_thresh = cv2.threshold(warped_gray, 180, 255, cv2.THRESH_BINARY)

    # Yöntem 2: Otsu Threshold
    _, otsu_thresh = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Yöntem 3: Gamma Düzeltme + Otsu
    warped_float = warped_gray.astype(np.float32) / 255.0
    warped_gamma = cv2.pow(warped_float, 0.8)
    gamma_corrected = (warped_gamma * 255).astype(np.uint8)
    _, gamma_thresh = cv2.threshold(gamma_corrected, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Yöntem 4: CLAHE + Otsu
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    clahe_applied = clahe.apply(warped_gray)
    _, thresh_clahe = cv2.threshold(clahe_applied, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 9. Gürültü Temizleme
    kernel = np.ones((2,2), np.uint8)
    cleaned = cv2.morphologyEx(thresh_clahe, cv2.MORPH_OPEN, kernel)

    # Görselleştirme: Kontrast ve Temizleme Yöntemleri
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 3, 1); plt.title("Perspektif Düzeltilmiş"); plt.imshow(warped_rgb); plt.axis('off')
    plt.subplot(2, 3, 2); plt.title("Basit Threshold"); plt.imshow(bin_thresh, cmap='gray'); plt.axis('off')
    plt.subplot(2, 3, 3); plt.title("Otsu Threshold"); plt.imshow(otsu_thresh, cmap='gray'); plt.axis('off')
    plt.subplot(2, 3, 4); plt.title("Gamma + Otsu"); plt.imshow(gamma_thresh, cmap='gray'); plt.axis('off')
    plt.subplot(2, 3, 5); plt.title("CLAHE + Otsu"); plt.imshow(thresh_clahe, cmap='gray'); plt.axis('off')
    plt.subplot(2, 3, 6); plt.title("Gürültü Temizlenmiş"); plt.imshow(cleaned, cmap='gray'); plt.axis('off')
    plt.tight_layout(); plt.savefig('results/Bolum1/step4_contrast_and_cleaning.png'); plt.close()

    return cleaned  # Nihai temizlenmiş görüntü döndürülür

# Örnek kullanım
if __name__ == "__main__":
    preprocess_document("images/evrak.jpg")