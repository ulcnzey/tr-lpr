"""Klasör bazlı toplu değerlendirme.

Kullanım:
  python -m pytest tests/test_batch_eval.py -v
  veya
  python tests/test_batch_eval.py path/to/images [--labels labels.csv]

labels.csv (opsiyonel): filename,plate
Örnek: car1.jpg,34ABC123

Görüntü yoksa testler atlanır — CI'yi kırmaz.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pipeline import Status, recognize_image

SAMPLE_DIR = ROOT / "samples"
REPORT_DIR = ROOT / "static" / "eval_reports"


@dataclass
class EvalRow:
    filename: str
    expected_plate: Optional[str]
    predicted_plate: Optional[str]
    status: str
    detection_ok: bool
    exact_match: bool
    processing_time_ms: float
    failure_reason: str


def load_labels(path: Path) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    if not path.exists():
        return labels
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("filename") or "").strip()
            plate = (row.get("plate") or "").strip().upper().replace(" ", "")
            if name:
                labels[name] = plate
    return labels


def evaluate_folder(
    image_dir: Path,
    labels: Optional[Dict[str, str]] = None,
) -> List[EvalRow]:
    labels = labels or {}
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    files: List[Path] = []
    for pat in patterns:
        files.extend(sorted(image_dir.glob(pat)))
        files.extend(sorted(image_dir.glob(pat.upper())))
    # unique
    files = sorted({f.resolve() for f in files})

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows: List[EvalRow] = []

    for path in files:
        image = cv2.imread(str(path))
        result = recognize_image(image)
        expected = labels.get(path.name)
        predicted = result.plate_text
        detection_ok = result.status != Status.PLATE_NOT_DETECTED
        exact = bool(expected and predicted and expected == predicted)

        rows.append(
            EvalRow(
                filename=path.name,
                expected_plate=expected,
                predicted_plate=predicted,
                status=result.status.value,
                detection_ok=detection_ok,
                exact_match=exact,
                processing_time_ms=result.processing_time_ms,
                failure_reason="" if result.is_valid else result.message,
            )
        )
    return rows


def summarize(rows: List[EvalRow]) -> dict:
    n = len(rows)
    if n == 0:
        return {"count": 0}
    labeled = [r for r in rows if r.expected_plate]
    detection_rate = sum(1 for r in rows if r.detection_ok) / n
    exact_rate = (
        sum(1 for r in labeled if r.exact_match) / len(labeled) if labeled else None
    )
    # Yanlış pozitif: geçerli denmiş ama etiket varsa ve eşleşmiyor
    fp = sum(
        1
        for r in labeled
        if r.predicted_plate and r.expected_plate and r.predicted_plate != r.expected_plate
    )
    fn = sum(1 for r in labeled if r.expected_plate and not r.predicted_plate)
    avg_ms = sum(r.processing_time_ms for r in rows) / n
    return {
        "count": n,
        "labeled_count": len(labeled),
        "detection_rate": round(detection_rate, 4),
        "exact_match_accuracy": None if exact_rate is None else round(exact_rate, 4),
        "false_positive_count": fp,
        "false_negative_count": fn,
        "avg_processing_time_ms": round(avg_ms, 2),
    }


@pytest.mark.skipif(not SAMPLE_DIR.exists(), reason="samples/ klasörü yok")
def test_batch_samples_folder_runs():
    rows = evaluate_folder(SAMPLE_DIR)
    assert isinstance(rows, list)
    # Klasör boş olabilir; o zaman da geçsin
    summary = summarize(rows)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "last_summary.json"
    report_path.write_text(
        json.dumps({"summary": summary, "rows": [asdict(r) for r in rows]}, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="TR-LPR klasör değerlendirmesi")
    parser.add_argument("image_dir", type=Path, help="Fotoğraf klasörü")
    parser.add_argument("--labels", type=Path, default=None, help="CSV: filename,plate")
    args = parser.parse_args()

    if not args.image_dir.is_dir():
        print(f"Klasör bulunamadı: {args.image_dir}")
        sys.exit(1)

    labels = load_labels(args.labels) if args.labels else {}
    t0 = time.perf_counter()
    rows = evaluate_folder(args.image_dir, labels)
    summary = summarize(rows)
    elapsed = (time.perf_counter() - t0) * 1000.0
    summary["wall_time_ms"] = round(elapsed, 2)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "last_summary.json"
    out.write_text(
        json.dumps({"summary": summary, "rows": [asdict(r) for r in rows]}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Rapor: {out}")


if __name__ == "__main__":
    main()
