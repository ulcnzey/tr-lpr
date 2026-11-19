"""Hafif OCR motoru — yalnızca kırpılmış plaka görüntüsünde çalışır.

RapidOCR (ONNX, CPU) tercih edilir. Motor süreç boyunca bir kez yüklenir.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from app import config


@dataclass(frozen=True)
class OcrVariantResult:
    variant: str
    text: str
    confidence: Optional[float]


@dataclass(frozen=True)
class OcrResult:
    text: str
    confidence: Optional[float]
    variants: Tuple[OcrVariantResult, ...]
    engine: str
    init_time_ms: float
    infer_time_ms: float
    message: str


class PlateOCR:
    """Tekil OCR sarmalayıcı — lazy init, yeniden kullanılabilir."""

    def __init__(self) -> None:
        self._engine = None
        self._engine_name = "none"
        self._init_time_ms = 0.0
        self._init_error: Optional[str] = None

    @property
    def ready(self) -> bool:
        return self._engine is not None

    @property
    def init_time_ms(self) -> float:
        return self._init_time_ms

    @property
    def engine_name(self) -> str:
        return self._engine_name

    def initialize(self) -> None:
        if self._engine is not None:
            return
        t0 = time.perf_counter()
        errors: list[str] = []
        # Önce unified `rapidocr`, yoksa legacy `rapidocr_onnxruntime`
        for module_name, engine_label in (
            ("rapidocr", "rapidocr"),
            ("rapidocr_onnxruntime", "rapidocr_onnxruntime"),
        ):
            try:
                module = __import__(module_name, fromlist=["RapidOCR"])
                self._engine = module.RapidOCR()
                self._engine_name = engine_label
                errors.clear()
                break
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{module_name}: {exc}")
        if self._engine is None:
            self._init_error = " | ".join(errors) if errors else "unknown"
            self._engine_name = "unavailable"
        self._init_time_ms = (time.perf_counter() - t0) * 1000.0

    def _preprocess_variants(self, plate_bgr: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """Az sayıda kontrollü varyant — aşırı büyütme yok."""
        variants: List[Tuple[str, np.ndarray]] = []
        if plate_bgr is None or plate_bgr.size == 0:
            return variants

        h, w = plate_bgr.shape[:2]
        # Hedef genişlik ~320; gereksiz upsample yok
        if w < 200:
            scale = 200 / float(w)
            plate_bgr = cv2.resize(
                plate_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC
            )
        elif w > 480:
            scale = 480 / float(w)
            plate_bgr = cv2.resize(
                plate_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
            )

        gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY) if plate_bgr.ndim == 3 else plate_bgr
        variants.append(("gray", gray))

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        variants.append(("clahe", clahe.apply(gray)))

        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(("otsu", otsu))

        return variants

    def _parse_engine_output(self, raw) -> List[Tuple[str, float]]:
        """rapidocr / rapidocr_onnxruntime çıktı biçimlerini normalize eder."""
        parsed: List[Tuple[str, float]] = []
        if raw is None:
            return parsed

        # Yeni rapidocr: nesne .txts / .scores
        if hasattr(raw, "txts") and raw.txts is not None:
            texts = list(raw.txts)
            scores = list(getattr(raw, "scores", []) or [None] * len(texts))
            for txt, score in zip(texts, scores):
                if txt:
                    parsed.append((str(txt).strip(), float(score) if score is not None else 0.0))
            return parsed

        # (result, elapse) tuple
        if isinstance(raw, tuple) and len(raw) >= 1:
            raw = raw[0]
        if not raw:
            return parsed

        for item in raw:
            if item is None:
                continue
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                # [box, text, score]
                txt = str(item[1]).strip()
                try:
                    score = float(item[2])
                except (TypeError, ValueError):
                    score = 0.0
                if txt:
                    parsed.append((txt, score))
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                # use_det=False: [text, score]
                txt = str(item[0]).strip()
                try:
                    score = float(item[1])
                except (TypeError, ValueError):
                    score = 0.0
                if txt:
                    parsed.append((txt, score))
            elif isinstance(item, dict):
                txt = str(item.get("text") or item.get("transcription") or "").strip()
                score = float(item.get("score") or item.get("confidence") or 0.0)
                if txt:
                    parsed.append((txt, score))
        return parsed

    def _run_engine(self, image: np.ndarray) -> Tuple[str, Optional[float]]:
        if self._engine is None:
            return "", None
        # Plaka zaten kırpıldığı için tespit/cls kapalı — CPU maliyeti düşer.
        try:
            raw = self._engine(image, use_det=False, use_cls=False)
        except TypeError:
            raw = self._engine(image)
        pairs = self._parse_engine_output(raw)
        texts: List[str] = []
        scores: List[float] = []
        for txt, score in pairs:
            if score < config.OCR_MIN_CONFIDENCE:
                continue
            texts.append(txt)
            scores.append(score)
        if not texts:
            return "", None
        # Birden fazla satır/kutu birleştirilmez — en yüksek skorlu tek metin.
        best_idx = int(np.argmax(scores))
        return texts[best_idx], scores[best_idx]

    def recognize(
        self,
        plate_bgr: np.ndarray,
        accept_fn=None,
    ) -> OcrResult:
        """OCR çalıştırır. accept_fn(text)->bool verilirse ilk kabul edilen varyantta durur."""
        self.initialize()
        if self._engine is None:
            return OcrResult(
                text="",
                confidence=None,
                variants=(),
                engine=self._engine_name,
                init_time_ms=self._init_time_ms,
                infer_time_ms=0.0,
                message=f"OCR motoru yüklenemedi: {self._init_error}",
            )

        t0 = time.perf_counter()
        variant_results: List[OcrVariantResult] = []
        best_text = ""
        best_conf: Optional[float] = None
        accepted = False

        for name, img in self._preprocess_variants(plate_bgr):
            text, conf = self._run_engine(img)
            variant_results.append(OcrVariantResult(variant=name, text=text, confidence=conf))
            if not text:
                continue
            if best_conf is None or (conf or 0) > (best_conf or 0):
                best_text = text
                best_conf = conf
            if accept_fn is not None and accept_fn(text):
                best_text = text
                best_conf = conf
                accepted = True
                break

        infer_ms = (time.perf_counter() - t0) * 1000.0
        msg = "OCR tamamlandı." if best_text else "OCR metin üretemedi."
        if accepted:
            msg = "OCR tamamlandı (erken durdu)."

        return OcrResult(
            text=best_text,
            confidence=best_conf,
            variants=tuple(variant_results),
            engine=self._engine_name,
            init_time_ms=self._init_time_ms,
            infer_time_ms=infer_ms,
            message=msg,
        )


# Süreç genelinde tek örnek
_OCR_SINGLETON: Optional[PlateOCR] = None


def get_ocr() -> PlateOCR:
    global _OCR_SINGLETON
    if _OCR_SINGLETON is None:
        _OCR_SINGLETON = PlateOCR()
    return _OCR_SINGLETON
