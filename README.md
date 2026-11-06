# Makine Öğrenmesi ile Meyve Ayıklama Simülasyonu

Elma, armut ve muzu renk ve şekil özelliklerine göre sınıflandıran bir Random Forest modeli ve sanal konveyör bant simülasyonu. Fiziksel kamera yerine test veri setindeki resimler kullanılmıştır.

## Kurulum (Python 3.10 veya üstü)

    pip install -r requirements.txt

## Çalıştırma

Simülasyonu başlat (eğitilmiş model ve veri repoda hazır):

    python simulate.py

Pencereyi kapatmak için `q` tuşuna bas. Farklı meyve sayısı için: `python simulate.py 50`

Modeli yeniden eğitip doğruluğu görmek için:

    python train.py

## Dosyalar

- `features.py`: Resimden renk (HSV histogramı) ve şekil özelliklerini çıkarır
- `train.py`: Modeli eğitir, test doğruluğunu ve confusion matrix'i üretir
- `simulate.py`: Sanal bant simülasyonu
- `model.joblib`: Eğitilmiş model
- `prepare_data.py`: `data/` klasörünü Fruits-360 veri setinden oluşturmak için kullanıldı (veri repoda hazır, çalıştırmak gerekmez)
- `data/`: Sınıf başına 600 eğitim, 200 test resmi. Kaynak: Fruits-360 (Mihai Oltean)