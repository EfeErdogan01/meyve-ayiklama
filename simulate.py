import random
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np

from features import extract

KOK = Path(__file__).parent
W, H = 1000, 520
BELT_Y = 160          # bandin ust cizgisi
FRUIT = 70            # ekrandaki meyve boyutu
CAMERA_X = 300        # kamera cizgisi
SPEED = 4             # bant hizi (piksel/kare)
BIN_X = {"apple": 520, "pear": 680, "banana": 840}
BIN_TOP, BIN_BOTTOM, BIN_HALF = 340, 480, 65
ISIM = {"apple": "ELMA", "pear": "ARMUT", "banana": "MUZ"}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def oku(yol):
    return cv2.imdecode(np.fromfile(str(yol), dtype=np.uint8), cv2.IMREAD_COLOR)


def yapistir(tuval, img, x, y):
    h, w = img.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, tuval.shape[1]), min(y + h, tuval.shape[0])
    if x0 >= x1 or y0 >= y1:
        return
    tuval[y0:y1, x0:x1] = img[y0 - y:y1 - y, x0 - x:x1 - x]


def main():
    adet = int(sys.argv[1]) if len(sys.argv) > 1 else 30

    paket = joblib.load(KOK / "model.joblib")
    model = paket["model"]
    model.n_jobs = 1  # tek resim icin daha hizli tahmin

    havuz = []
    for sinif in paket["classes"]:
        for yol in (KOK / "data" / "test" / sinif).glob("*.jpg"):
            havuz.append((yol, sinif))
    if not havuz:
        raise SystemExit("data/test icinde resim yok. Once prepare_data.py calistir.")

    meyveler = []
    sayac = {c: 0 for c in BIN_X}
    dogru = yanlis = uretilen = kare = 0

    while True:
        # Bantin soluna yeni meyve koy
        if uretilen < adet and kare % 45 == 0:
            yol, gercek = random.choice(havuz)
            img = oku(yol)
            meyveler.append({
                "img": img,
                "kucuk": cv2.resize(img, (FRUIT, FRUIT)),
                "x": -FRUIT, "y": BELT_Y - FRUIT,
                "gercek": gercek, "tahmin": None, "durum": "bant",
            })
            uretilen += 1

        # Meyveleri hareket ettir
        for f in meyveler:
            if f["durum"] == "bant":
                f["x"] += SPEED
                merkez = f["x"] + FRUIT // 2
                if f["tahmin"] is None and merkez >= CAMERA_X:
                    # Kamera cizgisi: model karar veriyor
                    f["tahmin"] = model.predict([extract(f["img"])])[0]
                if f["tahmin"] is not None and merkez >= BIN_X[f["tahmin"]]:
                    f["x"] = BIN_X[f["tahmin"]] - FRUIT // 2
                    f["durum"] = "dusuyor"
            elif f["durum"] == "dusuyor":
                f["y"] += 8
                if f["y"] + FRUIT >= BIN_BOTTOM:
                    f["durum"] = "bitti"
                    sayac[f["tahmin"]] += 1
                    if f["tahmin"] == f["gercek"]:
                        dogru += 1
                    else:
                        yanlis += 1
        meyveler = [f for f in meyveler if f["durum"] != "bitti"]

        # Cizim
        tuval = np.full((H, W, 3), 245, np.uint8)
        cv2.rectangle(tuval, (0, BELT_Y), (900, BELT_Y + 18), (70, 70, 70), -1)
        kayma = (kare * SPEED) % 30
        for x in range(0, 900, 30):
            cv2.line(tuval, (x + kayma, BELT_Y), (x + kayma, BELT_Y + 18), (110, 110, 110), 1)
        cv2.line(tuval, (CAMERA_X, 60), (CAMERA_X, BELT_Y + 18), (0, 0, 200), 2)
        cv2.putText(tuval, "KAMERA", (CAMERA_X - 40, 52), FONT, 0.6, (0, 0, 200), 2)

        for c, bx in BIN_X.items():
            cv2.rectangle(tuval, (bx - BIN_HALF, BIN_TOP), (bx + BIN_HALF, BIN_BOTTOM),
                          (120, 90, 60), 3)
            cv2.putText(tuval, f"{ISIM[c]}: {sayac[c]}", (bx - BIN_HALF + 8, BIN_BOTTOM + 30),
                        FONT, 0.7, (40, 40, 40), 2)

        for f in meyveler:
            yapistir(tuval, f["kucuk"], int(f["x"]), int(f["y"]))
            if f["tahmin"] is not None:
                renk = (0, 150, 0) if f["tahmin"] == f["gercek"] else (0, 0, 220)
                cv2.putText(tuval, ISIM[f["tahmin"]], (int(f["x"]), int(f["y"]) - 6),
                            FONT, 0.5, renk, 2)

        toplam = dogru + yanlis
        oran = 100 * dogru / toplam if toplam else 0
        cv2.putText(tuval,
                    f"Meyve: {uretilen}/{adet}   Dogru: {dogru}   Yanlis: {yanlis}   Basari: {oran:.1f}%",
                    (20, 30), FONT, 0.65, (0, 0, 0), 2)
        cv2.imshow("Meyve Ayiklama Simulasyonu (kapatmak icin q)", tuval)

        if cv2.waitKey(20) & 0xFF in (ord("q"), 27):
            break
        kare += 1
        if uretilen >= adet and not meyveler:
            cv2.waitKey(3000)
            break

    cv2.destroyAllWindows()
    toplam = dogru + yanlis
    if toplam:
        print(f"Toplam {toplam} meyve siniflandirildi: {dogru} dogru, {yanlis} yanlis, "
              f"basari %{100 * dogru / toplam:.1f}")
        print("Kutular:", {ISIM[c]: n for c, n in sayac.items()})


if __name__ == "__main__":
    main()