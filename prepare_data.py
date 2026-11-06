import random
import shutil
from pathlib import Path

KAYNAK = Path.home() / "Downloads" / "fruits-360-100x100-main" / "fruits-360-100x100-main"
SINIFLAR = {"apple": "Apple", "pear": "Pear", "banana": "Banana"}
LIMIT = {"Training": 600, "Test": 200}  # sinif basina resim sayisi
random.seed(42)

if not (KAYNAK / "Training").is_dir():
    raise SystemExit(f"Training bulunamadi: {KAYNAK}")

cikis = Path(__file__).parent / "data"
if cikis.exists():
    shutil.rmtree(cikis)

for bolum, limit in LIMIT.items():
    for sinif, onek in SINIFLAR.items():
        dosyalar = []
        for alt in (KAYNAK / bolum).iterdir():
            if alt.is_dir() and alt.name.startswith(onek):
                dosyalar += list(alt.glob("*.jpg"))
        random.shuffle(dosyalar)
        secilen = dosyalar[:limit]
        hedef = cikis / bolum.lower() / sinif
        hedef.mkdir(parents=True)
        for i, d in enumerate(secilen):
            shutil.copy(d, hedef / f"{i}_{d.name}")
        print(f"{bolum:9} {sinif:7}: {len(secilen)} resim kopyalandi (toplam bulunan: {len(dosyalar)})")

print("Bitti. Cikti klasoru:", cikis)