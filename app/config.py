"""Uygulama ayarları ve işlem sınırları."""

from pathlib import Path #/, \ ayrımıyla uğraşmadan) güvenli dosya ve klasör yolu oluşturmayı sağlar.

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
MODELS_DIR = BASE_DIR / "models"

# Dosya / istek sınırları bellek taşmasını önlemek adına kullanıyoruz
MAX_UPLOAD_BYTES = 8 * 1024 * 1024  # 8 MB
MAX_BATCH_FILES = 32 # toplu klasör inceleemsinde maksimum kabul edilen dosya sayısı
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_IMAGE_SIDE = 1600  # İşleme öncesi uzun kenar üst sınırı (OpenCV yolu)

# Eğitilmiş plaka dedektörü (Faster R-CNN / MobileNetV3-FPN)
# Not: dosya adı efficientdet olsa da checkpoint Faster R-CNN ağırlıklarıdır.
USE_TORCH_DETECTOR = True
PLATE_DETECTOR_WEIGHTS = MODELS_DIR / "plate_detector.pth"
DETECTOR_SCORE_THRESHOLD = 0.35
CROP_PAD_RATIO = 0.04

# OpenCV fallback tespit
MIN_PLATE_ASPECT = 2.0
MAX_PLATE_ASPECT = 6.5
MIN_PLATE_AREA_RATIO = 0.001
MAX_PLATE_AREA_RATIO = 0.25
MAX_CANDIDATES = 8

# OCR
OCR_LANGS = ["en"]  # Plaka karakterleri Latin; TR diacritic beklenmez
OCR_MIN_CONFIDENCE = 0.35


# API
API_PREFIX = "/api/v1"
HOST = "0.0.0.0"
PORT = 8000
