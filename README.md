# TR-LPR — Hafif Türkiye Plaka Tanıma Sistemi

CPU üzerinde çalışan, YOLO ve bulut OCR kullanmayan plaka tanıma backend'i.

**Akış:** plaka bölgesi tespiti → kırpma → OCR (yalnızca kırpılmış alan) → Türkiye plaka doğrulama.

## Mimari

| Modül | Görev |
|--------|--------|
| `app/detector.py` | Faster R-CNN plaka tespiti (+ OpenCV fallback) + kırpma |
| `app/ocr.py` | RapidOCR (ONNX, CPU), tek init |
| `app/validator.py` | TR sivil plaka format doğrulama |
| `app/pipeline.py` | Uçtan uca koordinasyon |
| `app/main.py` | Web UI + REST API |
| `run.py` | Sunucu başlatıcı |

## Kurulum

**Python 3.10–3.12** önerilir (`onnxruntime` tekerlekleri). Sistem varsayılanı 3.15 ise:

```bash
py -3.12 -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

## Çalıştırma

```bash
python run.py
```

Tarayıcı: [http://127.0.0.1:8000](http://127.0.0.1:8000)

Port meşgulse `app/config.py` içindeki `PORT` değerini değiştirin.

## API (mobil hazır)

### `GET /api/v1/health`

Servis ve OCR durumu.

### `POST /api/v1/recognize`

`multipart/form-data`, alan: `file`  
Opsiyonel query: `include_images=true` (base64; varsayılan kapalı)

Örnek yanıt:

```json
{
  "status": "plate_detected",
  "plate_text": "34ABC123",
  "is_valid": true,
  "confidence": 0.91,
  "processing_time_ms": 125.4,
  "bounding_box": {"x": 100, "y": 200, "width": 300, "height": 80},
  "message": "Geçerli standart sivil Türkiye plakası.",
  "raw_ocr": "34ABC123",
  "timing": {
    "detect_ms": 40.1,
    "ocr_ms": 70.2,
    "validate_ms": 0.3,
    "total_ms": 125.4,
    "ocr_init_ms": 800.0
  }
}
```

### `POST /api/v1/recognize/batch`

`files`: birden fazla görüntü.

### Durum kodları

| status | Anlam |
|--------|--------|
| `plate_detected` | Geçerli TR plaka |
| `plate_not_detected` | Aday bölge yok |
| `ocr_failed` | Kırpılmış alanda metin yok |
| `invalid_plate` | OCR çıktı ama format uymuyor |
| `uncertain` | Belirsiz (ileride genişletilebilir) |

Geçersiz OCR çıktısı başarılı plaka olarak **dönülmez**.

## Plaka doğrulama

Karayolları Trafik Yönetmeliği Madde 55 — standart sivil düzenler:

- 1 harf + 4/5 rakam
- 2 harf + 3/4 rakam
- 3 harf + 2/3 rakam

İl kodu: **01–81**. Q/W/X ve Türkçe diacritic harfler reddedilir.  
Firma/bayi/otomotiv birleşik metinleri reddedilir.  
O/0, I/1 vb. yalnızca **konum kuralına** göre düzeltilir; eksik karakter uydurulmaz.

> Not: Örneklerde geçen `01 AA 12` (2 harf + 2 rakam) resmi altı düzen arasında değildir; sistem bunu standart sivil olarak kabul etmez.

## Testler

```bash
python -m pytest tests/test_validator.py -v
python -m pytest tests/test_perf.py -v -s
```

Toplu fotoğraf değerlendirme:

```bash
# samples/ altına gerçek araç fotoğrafları koyun
python tests/test_batch_eval.py samples --labels samples/labels.csv
```

`labels.csv` formatı: `filename,plate`

Rapor: `static/eval_reports/last_summary.json`  
Performans: `static/eval_reports/perf_last.json`

## Plaka tespit modeli

`models/plate_detector.pth` — eğitilmiş plaka dedektörü.

Dosya adı `efficientdet_d0_plaka.pth` olsa da checkpoint içeriği **Faster R-CNN + MobileNetV3-Large FPN** (2 sınıf: arka plan + plaka). Model bir kez yüklenir; bulunamazsa OpenCV yoluna düşülür.

```text
models/plate_detector.pth   # Desktop'taki efficientdet_d0_plaka.pth kopyası
```

## Tasarım kararları

1. OCR asla tüm araç fotoğrafında çalışmaz.
2. Birden fazla OCR kutusundaki bağımsız yazılar birleştirilmez; en yüksek skorlu tek satır alınır, doğrulama eşiği uygulanır.
3. Yanlış plaka göstermek yerine `plate_not_detected` / `invalid_plate` tercih edilir.
4. OCR modeli süreçte bir kez yüklenir (`startup` + singleton).
5. Görüntü uzun kenarı `MAX_IMAGE_SIDE` ile sınırlanır; yükleme boyutu limiti vardır.

## Bilinen sınırlamalar

- Klasik OpenCV tespiti zor ışık, aşırı açı, ağır bulanıklık ve karmaşık arka planda başarısız olabilir.
- İlk sürüm özel/isim plakalarını standart sivil olarak **kabul etmez**.
- RapidOCR Latin karakter odaklıdır; plaka dışı Türkçe metin zaten kapsama alınmamalıdır.
- Gerçek doğruluk metrikleri `samples/` ile ölçülmelidir; sentetik görüntü yalnızca yol testi içindir.
- İleride hafif, özel eğitilmiş bir plaka dedektörü `detector.py` arkasından takılabilir; YOLO bu projede kullanılmaz.

## Kaynak hedefleri

- CPU only
- Tek worker (`run.py`)
- OCR lazy/singleton init
- Web sonuç görselleri diske yazılmaz; yalnızca tarayıcıya (base64) gönderilir
