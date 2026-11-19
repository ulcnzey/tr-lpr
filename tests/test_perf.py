"""CPU / RAM ve işlem süresi ölçümü (gerçek ölçüm)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

psutil = pytest.importorskip("psutil")

from app.detector import detect_plate
from app.ocr import get_ocr
from app.pipeline import recognize_image
from app.validator import validate_plate


def _synthetic_car_like(w: int = 800, h: int = 450) -> np.ndarray:
    """Gerçek fotoğraf yoksa detector/pipeline yolu için sentetik görüntü."""
    img = np.full((h, w, 3), 40, dtype=np.uint8)
    # Araç gövdesi benzeri
    cv2.rectangle(img, (80, 120), (w - 80, h - 40), (70, 70, 80), -1)
    # Plaka benzeri beyaz dikdörtgen
    x, y, pw, ph = w // 2 - 140, int(h * 0.72), 280, 70
    cv2.rectangle(img, (x, y), (x + pw, y + ph), (230, 230, 230), -1)
    cv2.rectangle(img, (x, y), (x + pw, y + ph), (20, 20, 20), 2)
    cv2.putText(
        img,
        "34 ABC 123",
        (x + 20, y + 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (10, 10, 10),
        2,
        cv2.LINE_AA,
    )
    return img


def test_validator_is_fast():
    t0 = time.perf_counter()
    for _ in range(1000):
        validate_plate("34 ABC 123")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 500.0


def test_detect_and_pipeline_timing_recorded():
    img = _synthetic_car_like()
    proc = psutil.Process()
    rss_before = proc.memory_info().rss

    t0 = time.perf_counter()
    det = detect_plate(img)
    detect_ms = (time.perf_counter() - t0) * 1000.0
    assert detect_ms > 0
    assert det.annotated is not None

    # OCR init ayrı ölçülsün
    ocr = get_ocr()
    t_init = time.perf_counter()
    ocr.initialize()
    init_ms = (time.perf_counter() - t_init) * 1000.0

    t1 = time.perf_counter()
    result = recognize_image(img)
    total_ms = (time.perf_counter() - t1) * 1000.0

    rss_after = proc.memory_info().rss
    cpu_percent = proc.cpu_percent(interval=0.1)

    metrics = {
        "detect_ms": round(detect_ms, 2),
        "ocr_init_ms": round(ocr.init_time_ms or init_ms, 2),
        "pipeline_ms": round(total_ms, 2),
        "pipeline_timing": {
            "detect_ms": round(result.timing.detect_ms, 2),
            "ocr_ms": round(result.timing.ocr_ms, 2),
            "validate_ms": round(result.timing.validate_ms, 2),
            "total_ms": round(result.timing.total_ms, 2),
        },
        "rss_before_mb": round(rss_before / (1024 * 1024), 2),
        "rss_after_mb": round(rss_after / (1024 * 1024), 2),
        "rss_delta_mb": round((rss_after - rss_before) / (1024 * 1024), 2),
        "cpu_percent_sample": cpu_percent,
        "status": result.status.value,
        "plate_text": result.plate_text,
        "ocr_engine": ocr.engine_name,
        "ocr_ready": ocr.ready,
    }
    out = ROOT / "static" / "eval_reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / "perf_last.json").write_text(
        __import__("json").dumps(metrics, indent=2),
        encoding="utf-8",
    )
    # Gerçek ölçüm kaydedildi; OCR kurulumu yoksa yine de süreler yazılmış olmalı
    assert result.timing.total_ms >= 0
    print("\nPERF:", metrics)
