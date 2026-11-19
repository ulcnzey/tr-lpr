"""Plaka tespiti ve kırpma.

Öncelik: eğitilmiş Faster R-CNN (MobileNetV3-FPN) plaka dedektörü.
Dosya adı efficientdet olsa da ağırlıklar Faster R-CNN'dir.
Model yoksa / yüklenemezse OpenCV aday bölge yöntemine düşer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from app import config


@dataclass(frozen=True)
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

    def as_xyxy(self) -> Tuple[int, int, int, int]: #köşe koordinatlarına çevirir
        return self.x, self.y, self.x + self.width, self.y + self.height

    def clip(self, img_w: int, img_h: int) -> "BoundingBox": #negatif veya taşan koordinatları engeller. 
        x1 = max(0, self.x)
        y1 = max(0, self.y)
        x2 = min(img_w, self.x + self.width)
        y2 = min(img_h, self.y + self.height)
        return BoundingBox(x1, y1, max(0, x2 - x1), max(0, y2 - y1))


@dataclass
class PlateCandidate:
    box: BoundingBox
    score: float
    quad: Optional[np.ndarray] = None  # 4x2 perspektif köşeleri


@dataclass
class DetectionResult:
    found: bool
    candidates: List[PlateCandidate]
    best: Optional[PlateCandidate]
    cropped: Optional[np.ndarray]
    annotated: Optional[np.ndarray]
    message: str
    engine: str = "none"


# ---------------------------------------------------------------------------
# Faster R-CNN plaka dedektörü (tek init)
# ---------------------------------------------------------------------------

_TORCH_MODEL = None
_TORCH_DEVICE = None
_TORCH_INIT_ERROR: Optional[str] = None
_TORCH_INIT_MS = 0.0


def detector_status() -> dict:
    return {
        "engine": "fasterrcnn_mobilenet_v3" if _TORCH_MODEL is not None else "opencv_fallback",
        "model_ready": _TORCH_MODEL is not None,
        "model_path": str(config.PLATE_DETECTOR_WEIGHTS),
        "init_error": _TORCH_INIT_ERROR,
        "init_time_ms": round(_TORCH_INIT_MS, 2),
        "device": str(_TORCH_DEVICE) if _TORCH_DEVICE is not None else None,
    }


def _build_fasterrcnn():
    import torch
    from torchvision.models import mobilenet_v3_large
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.backbone_utils import BackboneWithFPN
    from torchvision.models.detection.rpn import AnchorGenerator

    backbone = mobilenet_v3_large(weights=None).features
    # Checkpoint: FPN girişleri 160 ve 960 kanal (layer 13 + 16)
    bb = BackboneWithFPN(
        backbone,
        return_layers={"13": "0", "16": "1"},
        in_channels_list=[160, 960],
        out_channels=256,
        extra_blocks=None,
    )
    # FPN + LastLevelMaxPool → 3 özellik haritası
    anchors = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),) * 3,
        aspect_ratios=((0.5, 1.0, 2.0),) * 3,
    )
    model = FasterRCNN(bb, num_classes=2, rpn_anchor_generator=anchors)
    return model


def initialize_torch_detector() -> bool:
    """Ağırlıkları bir kez yükler. Başarısız olursa OpenCV fallback kullanılır."""
    global _TORCH_MODEL, _TORCH_DEVICE, _TORCH_INIT_ERROR, _TORCH_INIT_MS
    if _TORCH_MODEL is not None:
        return True

    import time

    t0 = time.perf_counter()
    weights = Path(config.PLATE_DETECTOR_WEIGHTS)
    if not weights.is_file():
        _TORCH_INIT_ERROR = f"Ağırlık dosyası yok: {weights}"
        _TORCH_INIT_MS = (time.perf_counter() - t0) * 1000.0
        return False

    try:
        import torch
        from torchvision.transforms import functional as TF  # noqa: F401

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _build_fasterrcnn()
        state = torch.load(str(weights), map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        cleaned = {
            (k[7:] if k.startswith("module.") else k): v for k, v in state.items()
        }
        model.load_state_dict(cleaned, strict=True)
        model.to(device)
        model.eval()
        _TORCH_MODEL = model
        _TORCH_DEVICE = device
        _TORCH_INIT_ERROR = None
        _TORCH_INIT_MS = (time.perf_counter() - t0) * 1000.0
        return True
    except Exception as exc:  # noqa: BLE001
        _TORCH_MODEL = None
        _TORCH_DEVICE = None
        _TORCH_INIT_ERROR = str(exc)
        _TORCH_INIT_MS = (time.perf_counter() - t0) * 1000.0
        return False


def _detect_with_torch(image: np.ndarray) -> List[PlateCandidate]:
    import torch
    from torchvision.transforms import functional as TF

    assert _TORCH_MODEL is not None and _TORCH_DEVICE is not None
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = TF.to_tensor(rgb).to(_TORCH_DEVICE)
    with torch.inference_mode():
        out = _TORCH_MODEL([tensor])[0]

    boxes = out["boxes"].detach().cpu().numpy()
    scores = out["scores"].detach().cpu().numpy()
    labels = out["labels"].detach().cpu().numpy()

    candidates: List[PlateCandidate] = []
    h, w = image.shape[:2]
    for box, score, label in zip(boxes, scores, labels):
        if int(label) != 1:
            continue
        if float(score) < config.DETECTOR_SCORE_THRESHOLD:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        bb = BoundingBox(x1, y1, max(0, x2 - x1), max(0, y2 - y1)).clip(w, h)
        if bb.width < 16 or bb.height < 8:
            continue
        aspect = bb.width / float(bb.height)
        if aspect < 1.5 or aspect > 8.0:
            continue
        candidates.append(PlateCandidate(box=bb, score=float(score), quad=None))

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[: config.MAX_CANDIDATES]


# ---------------------------------------------------------------------------
# OpenCV fallback
# ---------------------------------------------------------------------------

def resize_for_processing(image: np.ndarray, max_side: int = config.MAX_IMAGE_SIDE) -> Tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image, 1.0
    scale = max_side / float(longest)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def _score_candidate(
    box: BoundingBox,
    img_w: int,
    img_h: int,
    solidity: float,
    extent: float,
) -> float:
    if box.width <= 0 or box.height <= 0:
        return -1.0
    aspect = box.width / float(box.height)
    area = box.width * box.height
    img_area = float(img_w * img_h)
    area_ratio = area / img_area

    if aspect < config.MIN_PLATE_ASPECT or aspect > config.MAX_PLATE_ASPECT:
        return -1.0
    if area_ratio < config.MIN_PLATE_AREA_RATIO or area_ratio > config.MAX_PLATE_AREA_RATIO:
        return -1.0

    aspect_score = 1.0 - min(abs(aspect - 4.2) / 3.0, 1.0)
    cy = box.y + box.height / 2.0
    vertical_score = 1.0 - abs((cy / img_h) - 0.72) / 0.72
    vertical_score = max(0.0, min(1.0, vertical_score))
    area_score = 1.0 - abs(area_ratio - 0.03) / 0.03
    area_score = max(0.0, min(1.0, area_score))
    shape_score = 0.5 * solidity + 0.5 * extent

    return 0.35 * aspect_score + 0.25 * vertical_score + 0.20 * area_score + 0.20 * shape_score


def _morph_candidates(gray: np.ndarray) -> np.ndarray:
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(blur, 60, 180)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    dilated = cv2.dilate(closed, kernel, iterations=1)
    return dilated


def _threshold_candidates(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thr = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 7
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    return cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)


def _extract_from_binary(
    binary: np.ndarray,
    img_w: int,
    img_h: int,
) -> List[PlateCandidate]:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: List[PlateCandidate] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 400:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        x, y, w, h = cv2.boundingRect(cnt if len(approx) < 4 else approx)
        box = BoundingBox(int(x), int(y), int(w), int(h))
        rect_area = max(w * h, 1)
        solidity = float(area) / rect_area
        extent = float(area) / rect_area
        score = _score_candidate(box, img_w, img_h, solidity, extent)
        if score < 0.25:
            continue
        quad = None
        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype(np.float32)
        candidates.append(PlateCandidate(box=box, score=score, quad=quad))
    return candidates


def _nms(candidates: Sequence[PlateCandidate], iou_thresh: float = 0.35) -> List[PlateCandidate]:
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda c: c.score, reverse=True)
    kept: List[PlateCandidate] = []
    for cand in ordered:
        if all(_iou(cand.box, k.box) < iou_thresh for k in kept):
            kept.append(cand)
        if len(kept) >= config.MAX_CANDIDATES:
            break
    return kept


def _iou(a: BoundingBox, b: BoundingBox) -> float:
    ax1, ay1, ax2, ay2 = a.as_xyxy()
    bx1, by1, bx2, by2 = b.as_xyxy()
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = a.width * a.height + b.width * b.height - inter
    return inter / float(union) if union > 0 else 0.0


def _order_quad(pts: np.ndarray) -> np.ndarray:
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def warp_plate(image: np.ndarray, quad: np.ndarray, out_w: int = 320, out_h: int = 80) -> np.ndarray:
    ordered = _order_quad(quad.astype(np.float32))
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(ordered, dst)
    return cv2.warpPerspective(image, matrix, (out_w, out_h))


def crop_box(image: np.ndarray, box: BoundingBox, pad_ratio: float = 0.04) -> np.ndarray:
    h, w = image.shape[:2]
    pad_x = int(box.width * pad_ratio)
    pad_y = int(box.height * pad_ratio)
    x1 = max(0, box.x - pad_x)
    y1 = max(0, box.y - pad_y)
    x2 = min(w, box.x + box.width + pad_x)
    y2 = min(h, box.y + box.height + pad_y)
    return image[y1:y2, x1:x2].copy()


def annotate(image: np.ndarray, box: BoundingBox, color: Tuple[int, int, int] = (0, 180, 0)) -> np.ndarray:
    out = image.copy()
    x1, y1, x2, y2 = box.as_xyxy()
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    return out


def _detect_opencv(image: np.ndarray) -> Tuple[List[PlateCandidate], Optional[np.ndarray]]:
    working, scale = resize_for_processing(image)
    h, w = working.shape[:2]
    gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY) if working.ndim == 3 else working
    morph = _morph_candidates(gray)
    thr = _threshold_candidates(gray)
    raw_candidates = _extract_from_binary(morph, w, h) + _extract_from_binary(thr, w, h)
    candidates = _nms(raw_candidates)
    if not candidates:
        return [], working

    best = candidates[0]
    if scale != 1.0:
        inv = 1.0 / scale
        scaled = []
        for cand in candidates:
            box = BoundingBox(
                x=int(cand.box.x * inv),
                y=int(cand.box.y * inv),
                width=int(cand.box.width * inv),
                height=int(cand.box.height * inv),
            ).clip(image.shape[1], image.shape[0])
            quad = None
            if cand.quad is not None:
                quad = cand.quad * inv
            scaled.append(PlateCandidate(box=box, score=cand.score, quad=quad))
        return scaled, annotate(image, scaled[0].box)
    return candidates, annotate(image, best.box)


def detect_plate(image: np.ndarray) -> DetectionResult:
    """Plaka adaylarını tespit eder; en iyisini kırpar."""
    if image is None or image.size == 0:
        return DetectionResult(
            found=False,
            candidates=[],
            best=None,
            cropped=None,
            annotated=None,
            message="Geçersiz görüntü.",
            engine="none",
        )

    engine = "opencv"
    candidates: List[PlateCandidate] = []
    annotated: Optional[np.ndarray] = None

    if config.USE_TORCH_DETECTOR:
        if initialize_torch_detector():
            try:
                candidates = _detect_with_torch(image)
                engine = "fasterrcnn_mobilenet_v3"
                if candidates:
                    annotated = annotate(image, candidates[0].box)
            except Exception as exc:  # noqa: BLE001
                candidates = []
                engine = "opencv"
                annotated = None
                # sessizce fallback; mesajda belirtilecek
                fallback_reason = str(exc)
            else:
                fallback_reason = None
        else:
            fallback_reason = _TORCH_INIT_ERROR
    else:
        fallback_reason = "torch detector kapalı"

    if not candidates:
        candidates, annotated = _detect_opencv(image)
        engine = "opencv" if engine != "fasterrcnn_mobilenet_v3" else "opencv_after_empty_torch"
        if fallback_reason and not candidates:
            return DetectionResult(
                found=False,
                candidates=[],
                best=None,
                cropped=None,
                annotated=annotated,
                message=f"Plaka aday bölgesi bulunamadı. ({fallback_reason})",
                engine=engine,
            )

    if not candidates:
        return DetectionResult(
            found=False,
            candidates=[],
            best=None,
            cropped=None,
            annotated=annotated,
            message="Plaka aday bölgesi bulunamadı.",
            engine=engine,
        )

    best = candidates[0]
    if annotated is None:
        annotated = annotate(image, best.box)

    if best.quad is not None:
        cropped = warp_plate(image, best.quad)
    else:
        cropped = crop_box(image, best.box, pad_ratio=config.CROP_PAD_RATIO)

    if cropped is None or cropped.size == 0:
        return DetectionResult(
            found=False,
            candidates=candidates,
            best=None,
            cropped=None,
            annotated=annotated,
            message="Plaka kırpılamadı.",
            engine=engine,
        )

    return DetectionResult(
        found=True,
        candidates=candidates,
        best=best,
        cropped=cropped,
        annotated=annotated,
        message=f"Plaka adayı tespit edildi ({engine}).",
        engine=engine,
    )
