from pathlib import Path

import cv2
import joblib
import matplotlib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix)

from features import extract

matplotlib.use("Agg")
import matplotlib.pyplot as plt

KOK = Path(__file__).parent
SINIFLAR = ["apple", "pear", "banana"]


def yukle(bolum):
    X, y = [], []
    for etiket in SINIFLAR:
        for yol in sorted((KOK / "data" / bolum / etiket).glob("*.jpg")):
            img = cv2.imdecode(np.fromfile(str(yol), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            X.append(extract(img))
            y.append(etiket)
        print(f"  {bolum}/{etiket}: yuklendi")
    return np.array(X), np.array(y)


print("Veri yukleniyor...")
X_train, y_train = yukle("training")
X_test, y_test = yukle("test")
print(f"Egitim: {len(X_train)} resim, Test: {len(X_test)} resim, "
      f"Ozellik sayisi: {X_train.shape[1]}")

print("Model egitiliyor...")
model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

tahmin = model.predict(X_test)
print(f"\nTest dogrulugu: {accuracy_score(y_test, tahmin) * 100:.2f}%\n")
print(classification_report(y_test, tahmin, labels=SINIFLAR))

cm = confusion_matrix(y_test, tahmin, labels=SINIFLAR)
print("Confusion matrix (satir=gercek, sutun=tahmin):")
print(cm)

ConfusionMatrixDisplay(cm, display_labels=SINIFLAR).plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig(KOK / "confusion_matrix.png", dpi=150, bbox_inches="tight")

joblib.dump({"model": model, "classes": SINIFLAR}, KOK / "model.joblib")
print("\nModel kaydedildi: model.joblib")
print("Grafik kaydedildi: confusion_matrix.png")