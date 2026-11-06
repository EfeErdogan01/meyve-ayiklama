import cv2
import numpy as np


def extract(img_bgr):
    """Bir meyve resminden sayisal ozellik vektoru cikarir (renk + sekil)."""
    img = cv2.resize(img_bgr, (100, 100))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Beyaz olmayan pikseller = meyve (arka plan beyaz)
    mask = (img.min(axis=2) < 235).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # Sekil ozellikleri
    shape = [0.0, 0.0, 0.0, 0.0, 0.0]
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        per = cv2.arcLength(c, True)
        hull_area = cv2.contourArea(cv2.convexHull(c))
        (w, h) = cv2.minAreaRect(c)[1]
        if area > 0 and per > 0 and w > 0 and h > 0:
            aspect = max(w, h) / min(w, h)             # en/boy orani
            solidity = area / max(hull_area, 1e-6)     # ne kadar "dolgun"
            circularity = 4 * np.pi * area / (per * per)  # yuvarlaklik
            area_ratio = area / (100 * 100)            # resmi ne kadar kapliyor
            extent = area / (w * h)                    # kutuyu doldurma orani
            shape = [aspect, solidity, circularity, area_ratio, extent]

    # Renk ozellikleri: sadece meyve pikselleri uzerinden HSV histogrami
    m = mask if mask.sum() > 0 else None
    feats = []
    for ch, bins, rng in ((0, 18, [0, 180]), (1, 8, [0, 256]), (2, 8, [0, 256])):
        hist = cv2.calcHist([hsv], [ch], m, [bins], rng).flatten()
        feats += list(hist / (hist.sum() + 1e-6))

    return np.array(feats + shape, dtype=np.float32)