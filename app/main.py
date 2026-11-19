"""FastAPI uygulaması — web arayüzü ve mobil-hazır REST API."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app import config
from app.detector import detector_status, initialize_torch_detector
from app.ocr import get_ocr
from app.pipeline import Status, decode_image_bytes, recognize_bytes, recognize_image

config.STATIC_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="TR-LPR",
    description="Hafif Türkiye plaka tanıma API",
    version="0.1.0",
)

app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def _check_extension(filename: str) -> None:
    ext = Path(filename).suffix.lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Desteklenmeyen dosya türü: {ext}. İzin verilenler: {sorted(config.ALLOWED_EXTENSIONS)}",
        )


async def _read_upload(file: UploadFile) -> bytes:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail=f"Boş dosya: {file.filename}")
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"Dosya çok büyük (max {config.MAX_UPLOAD_BYTES} bayt): {file.filename}",
        )
    return data


def _encode_jpeg_b64(image: Optional[np.ndarray], quality: int = 80) -> Optional[str]:
    """Görseli diske yazmadan tarayıcıda göstermek için data-URL üretir."""
    if image is None:
        return None
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


@app.on_event("startup")
def startup() -> None:
    # Dedektör + OCR'yi istek öncesi bir kez yükle
    initialize_torch_detector()
    ocr = get_ocr()
    ocr.initialize()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get(f"{config.API_PREFIX}/health")
async def health():
    ocr = get_ocr()
    return {
        "status": "ok",
        "ocr_engine": ocr.engine_name,
        "ocr_ready": ocr.ready,
        "ocr_init_time_ms": round(ocr.init_time_ms, 2),
        "detector": detector_status(),
    }


@app.post(f"{config.API_PREFIX}/recognize")
async def api_recognize(
    file: UploadFile = File(...),
    include_images: bool = False,
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Dosya adı gerekli.")
    _check_extension(file.filename)
    data = await _read_upload(file)
    result = recognize_bytes(data)
    payload = result.to_api_dict()
    if include_images:
        payload["images"] = {
            "annotated_base64": _encode_jpeg_b64(result.annotated_image),
            "cropped_base64": _encode_jpeg_b64(result.cropped_image),
        }
    return JSONResponse(payload)


@app.post(f"{config.API_PREFIX}/recognize/batch")
async def api_recognize_batch(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="En az bir dosya gerekli.")
    if len(files) > config.MAX_BATCH_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"En fazla {config.MAX_BATCH_FILES} dosya yüklenebilir.",
        )
    results = []
    for f in files:
        name = f.filename or "unknown"
        try:
            _check_extension(name)
            data = await _read_upload(f)
            result = recognize_bytes(data)
            item = result.to_api_dict()
            item["filename"] = name
            results.append(item)
        except HTTPException as exc:
            results.append(
                {
                    "filename": name,
                    "status": Status.PLATE_NOT_DETECTED.value,
                    "plate_text": None,
                    "is_valid": False,
                    "confidence": None,
                    "processing_time_ms": 0,
                    "bounding_box": None,
                    "message": str(exc.detail),
                }
            )
    return {"count": len(results), "results": results}


@app.post("/web/recognize")
async def web_recognize(files: List[UploadFile] = File(...)):
    """Web arayüzü: görseller diske yazılmaz, yalnızca yanıtta (base64) döner."""
    if not files:
        raise HTTPException(status_code=400, detail="Dosya seçilmedi.")
    if len(files) > config.MAX_BATCH_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"En fazla {config.MAX_BATCH_FILES} dosya.",
        )

    out = []
    for f in files:
        name = f.filename or "unknown"
        try:
            _check_extension(name)
            data = await _read_upload(f)
            image = decode_image_bytes(data)
            result = recognize_image(image)
            out.append(
                {
                    "filename": name,
                    "data": result.to_api_dict(),
                    "images": {
                        "original": _encode_jpeg_b64(image),
                        "annotated": _encode_jpeg_b64(result.annotated_image),
                        "cropped": _encode_jpeg_b64(result.cropped_image),
                    },
                }
            )
        except HTTPException as exc:
            out.append(
                {
                    "filename": name,
                    "data": {
                        "status": Status.PLATE_NOT_DETECTED.value,
                        "plate_text": None,
                        "is_valid": False,
                        "confidence": None,
                        "processing_time_ms": 0,
                        "bounding_box": None,
                        "message": str(exc.detail),
                    },
                    "images": {},
                    "error": str(exc.detail),
                }
            )
        except Exception as exc:  # noqa: BLE001
            out.append(
                {
                    "filename": name,
                    "data": {
                        "status": Status.OCR_FAILED.value,
                        "plate_text": None,
                        "is_valid": False,
                        "confidence": None,
                        "processing_time_ms": 0,
                        "bounding_box": None,
                        "message": f"İşleme hatası: {exc}",
                    },
                    "images": {},
                    "error": str(exc),
                }
            )
    return {"results": out}
