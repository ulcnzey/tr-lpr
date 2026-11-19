"""Tespit → kırpma → OCR → doğrulama koordinasyonu."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from app.detector import BoundingBox, detect_plate
from app.ocr import get_ocr
from app.validator import ValidationResult, validate_plate


class Status(str, Enum):
    PLATE_DETECTED = "plate_detected"
    PLATE_NOT_DETECTED = "plate_not_detected"
    OCR_FAILED = "ocr_failed"
    INVALID_PLATE = "invalid_plate"
    UNCERTAIN = "uncertain"


@dataclass
class TimingInfo:
    detect_ms: float = 0.0
    ocr_ms: float = 0.0
    validate_ms: float = 0.0
    total_ms: float = 0.0
    ocr_init_ms: float = 0.0


@dataclass
class RecognizeResult:
    status: Status
    plate_text: Optional[str]
    is_valid: bool
    confidence: Optional[float]
    processing_time_ms: float
    bounding_box: Optional[Dict[str, int]]
    message: str
    raw_ocr: Optional[str] = None
    timing: TimingInfo = field(default_factory=TimingInfo)
    annotated_image: Optional[np.ndarray] = None
    cropped_image: Optional[np.ndarray] = None
    validation: Optional[ValidationResult] = None

    def to_api_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "plate_text": self.plate_text,
            "is_valid": self.is_valid,
            "confidence": self.confidence,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "bounding_box": self.bounding_box,
            "message": self.message,
            "raw_ocr": self.raw_ocr,
            "timing": {
                "detect_ms": round(self.timing.detect_ms, 2),
                "ocr_ms": round(self.timing.ocr_ms, 2),
                "validate_ms": round(self.timing.validate_ms, 2),
                "total_ms": round(self.timing.total_ms, 2),
                "ocr_init_ms": round(self.timing.ocr_init_ms, 2),
            },
        }


def decode_image_bytes(data: bytes) -> Optional[np.ndarray]:
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return image


def _box_dict(box: Optional[BoundingBox]) -> Optional[Dict[str, int]]:
    if box is None:
        return None
    return {"x": box.x, "y": box.y, "width": box.width, "height": box.height}


def recognize_image(image: np.ndarray) -> RecognizeResult:
    """Tek görüntü için uçtan uca tanıma."""
    t_total = time.perf_counter()
    timing = TimingInfo()

    if image is None or image.size == 0:
        timing.total_ms = (time.perf_counter() - t_total) * 1000.0
        return RecognizeResult(
            status=Status.PLATE_NOT_DETECTED,
            plate_text=None,
            is_valid=False,
            confidence=None,
            processing_time_ms=timing.total_ms,
            bounding_box=None,
            message="Görüntü okunamadı.",
            timing=timing,
        )

    t0 = time.perf_counter()
    detection = detect_plate(image)
    timing.detect_ms = (time.perf_counter() - t0) * 1000.0

    if not detection.found or detection.cropped is None or detection.best is None:
        timing.total_ms = (time.perf_counter() - t_total) * 1000.0
        return RecognizeResult(
            status=Status.PLATE_NOT_DETECTED,
            plate_text=None,
            is_valid=False,
            confidence=None,
            processing_time_ms=timing.total_ms,
            bounding_box=None,
            message=detection.message or "Plaka tespit edilemedi.",
            timing=timing,
            annotated_image=detection.annotated,
        )

    ocr = get_ocr()
    t1 = time.perf_counter()
    # Geçerli TR plaka bulununca OCR varyant döngüsünden çık
    ocr_result = ocr.recognize(
        detection.cropped,
        accept_fn=lambda text: validate_plate(text).is_valid,
    )
    timing.ocr_ms = (time.perf_counter() - t1) * 1000.0
    timing.ocr_init_ms = ocr_result.init_time_ms

    if not ocr_result.text:
        any_text = next((v.text for v in ocr_result.variants if v.text), "")
        if not any_text:
            timing.total_ms = (time.perf_counter() - t_total) * 1000.0
            return RecognizeResult(
                status=Status.OCR_FAILED,
                plate_text=None,
                is_valid=False,
                confidence=None,
                processing_time_ms=timing.total_ms,
                bounding_box=_box_dict(detection.best.box),
                message=ocr_result.message or "OCR başarısız.",
                raw_ocr=None,
                timing=timing,
                annotated_image=detection.annotated,
                cropped_image=detection.cropped,
            )

    t2 = time.perf_counter()
    best_validation: Optional[ValidationResult] = None
    best_conf: Optional[float] = None
    chosen_raw = ocr_result.text

    candidates = [(ocr_result.text, ocr_result.confidence)]
    for vr in ocr_result.variants:
        if vr.text and (vr.text, vr.confidence) not in candidates:
            candidates.append((vr.text, vr.confidence))

    for text, conf in candidates:
        validation = validate_plate(text)
        if validation.is_valid:
            best_validation = validation
            best_conf = conf
            chosen_raw = text
            break
        if best_validation is None:
            best_validation = validation
            best_conf = conf
            chosen_raw = text

    timing.validate_ms = (time.perf_counter() - t2) * 1000.0
    timing.total_ms = (time.perf_counter() - t_total) * 1000.0

    assert best_validation is not None

    if best_validation.is_valid:
        return RecognizeResult(
            status=Status.PLATE_DETECTED,
            plate_text=best_validation.plate_text,
            is_valid=True,
            confidence=best_conf,
            processing_time_ms=timing.total_ms,
            bounding_box=_box_dict(detection.best.box),
            message=best_validation.message,
            raw_ocr=chosen_raw,
            timing=timing,
            annotated_image=detection.annotated,
            cropped_image=detection.cropped,
            validation=best_validation,
        )

    # Geçersiz ama OCR çıktı verdi
    status = Status.INVALID_PLATE
    if best_validation.reject_reason and best_validation.reject_reason.value in {
        "ambiguous",
    }:
        status = Status.UNCERTAIN

    return RecognizeResult(
        status=status,
        plate_text=None,
        is_valid=False,
        confidence=best_conf,
        processing_time_ms=timing.total_ms,
        bounding_box=_box_dict(detection.best.box),
        message=best_validation.message,
        raw_ocr=chosen_raw,
        timing=timing,
        annotated_image=detection.annotated,
        cropped_image=detection.cropped,
        validation=best_validation,
    )


def recognize_bytes(data: bytes) -> RecognizeResult:
    image = decode_image_bytes(data)
    return recognize_image(image)


def recognize_batch(images: List[np.ndarray]) -> List[RecognizeResult]:
    return [recognize_image(img) for img in images]
