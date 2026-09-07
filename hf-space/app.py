"""
SCOS Snooker Ball Detection — Hugging Face Space

FastAPI server that loads a YOLOv8 ONNX model from Hugging Face Hub
and serves a /detect endpoint compatible with the SCOS DetectorAdapter contract.

Deploy:
  1. Create a new Space at https://huggingface.co/new-space (Docker type, CPU)
  2. Copy this file + requirements.txt + README.md
  3. Set HF_MODEL_REPO in Space Variables (or use default)
  4. Space URL: https://yourname-scoker-detector.hf.space
"""

from __future__ import annotations

import base64
import io
import os
import time
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# --- Configuration ---------------------------------------------------------

HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "your-username/scos-yolov8s")
MODEL_FILENAME = os.environ.get("MODEL_FILENAME", "best.pt")
CONFIDENCE_DEFAULT = float(os.environ.get("CONFIDENCE_DEFAULT", "0.25"))
IOU_THRESHOLD = float(os.environ.get("IOU_THRESHOLD", "0.45"))
IMGSZ = int(os.environ.get("IMGSZ", "640"))

# SCOS ball class names (normalized)
SCOS_CLASSES = [
    "red", "yellow", "green", "brown",
    "blue", "pink", "black", "cue_ball",
]

# Map numeric pool-ball class IDs (0-15) to SCOS names where possible
# Pool: 0=cue, 1-7=solids, 8-15=stripes
# Snooker mapping is approximate — adjust after training on snooker data
POOL_TO_SCOS: dict[int, str] = {
    0: "cue_ball",
    1: "red",
    2: "yellow",
    3: "green",
    4: "brown",
    5: "blue",
    6: "pink",
    7: "black",
}

app = FastAPI(title="SCOS Snooker Detector", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model: YOLO | None = None
_model_names: dict[int, str] = {}


def get_model() -> YOLO:
    global _model, _model_names
    if _model is not None:
        return _model

    t0 = time.perf_counter()
    model_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename=MODEL_FILENAME)
    _model = YOLO(model_path)
    _model_names = dict(enumerate(_model.names.values())) if _model.names else {}
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"[scos] Model loaded from {HF_MODEL_REPO}/{MODEL_FILENAME} in {load_ms:.0f}ms")
    print(f"[scos] Model classes: {_model_names}")
    return _model


def normalize_class(cls_id: int, raw_name: str) -> str:
    """Map model class ID/name to SCOS normalized class name."""
    key = str(cls_id)
    if key in POOL_TO_SCOS:
        return POOL_TO_SCOS[key]
    raw_lower = (raw_name or "").strip().lower()
    aliases = {
        "white": "cue_ball", "cue": "cue_ball", "cue_ball": "cue_ball",
        "red": "red", "reds": "red",
        "yellow": "yellow", "green": "green", "brown": "brown",
        "blue": "blue", "pink": "pink", "black": "black",
    }
    if raw_lower in aliases:
        return aliases[raw_lower]
    return raw_lower.replace(" ", "_") or "unknown"


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "model_repo": HF_MODEL_REPO,
        "model_file": MODEL_FILENAME,
        "model_loaded": _model is not None,
        "classes": _model_names if _model_names else {},
        "scos_classes": SCOS_CLASSES,
    }


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    confidence: float = Form(CONFIDENCE_DEFAULT),
    iou: float = Form(IOU_THRESHOLD),
):
    """
    Run ball detection on an uploaded JPEG image.

    Returns SCOS-normalized detections:
      { class, confidence, x, y, width, height, raw_class, model_class_id }

    x, y = top-left corner in source image pixels.
    """
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty image")

    t_decode = time.perf_counter()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode image")
    h, w = img.shape[:2]
    decode_ms = (time.perf_counter() - t_decode) * 1000

    model = get_model()

    t_infer = time.perf_counter()
    results = model(
        img,
        conf=confidence,
        iou=iou,
        imgsz=IMGSZ,
        verbose=False,
    )
    infer_ms = (time.perf_counter() - t_infer) * 1000

    r = results[0]
    detections = []

    if r.boxes is not None and len(r.boxes) > 0:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = xyxy
            raw_name = _model_names.get(cls_id, str(cls_id))
            scos_class = normalize_class(cls_id, raw_name)

            detections.append({
                "class": scos_class,
                "raw_class": raw_name,
                "model_class_id": cls_id,
                "confidence": round(conf, 4),
                "x": round(x1, 1),
                "y": round(y1, 1),
                "width": round(x2 - x1, 1),
                "height": round(y2 - y1, 1),
            })

    counts: dict[str, int] = {}
    for d in detections:
        counts[d["class"]] = counts.get(d["class"], 0) + 1

    return {
        "ok": True,
        "detections": detections,
        "stats": {
            "total": len(detections),
            "counts_by_class": counts,
        },
        "image_size": {"width": w, "height": h},
        "timings_ms": {
            "decode": round(decode_ms, 2),
            "inference": round(infer_ms, 2),
            "total": round(decode_ms + infer_ms, 2),
        },
        "model": {
            "repo": HF_MODEL_REPO,
            "file": MODEL_FILENAME,
            "imgsz": IMGSZ,
            "confidence_threshold": confidence,
            "iou_threshold": iou,
        },
    }


@app.post("/detect/base64")
async def detect_base64(payload: dict[str, Any]):
    """
    Alternative endpoint for base64-encoded images.
    Body: { "image": "data:image/jpeg;base64,...", "confidence": 0.25 }
    """
    image_data = payload.get("image", "")
    if not image_data:
        raise HTTPException(400, "image field required")

    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    try:
        raw = base64.b64decode(image_data)
    except Exception:
        raise HTTPException(400, "Invalid base64 image")

    confidence = float(payload.get("confidence", CONFIDENCE_DEFAULT))
    iou = float(payload.get("iou", IOU_THRESHOLD))

    t_decode = time.perf_counter()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode image")
    h, w = img.shape[:2]
    decode_ms = (time.perf_counter() - t_decode) * 1000

    model = get_model()

    t_infer = time.perf_counter()
    results = model(img, conf=confidence, iou=iou, imgsz=IMGSZ, verbose=False)
    infer_ms = (time.perf_counter() - t_infer) * 1000

    r = results[0]
    detections = []

    if r.boxes is not None and len(r.boxes) > 0:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = xyxy
            raw_name = _model_names.get(cls_id, str(cls_id))
            scos_class = normalize_class(cls_id, raw_name)

            detections.append({
                "class": scos_class,
                "raw_class": raw_name,
                "model_class_id": cls_id,
                "confidence": round(conf, 4),
                "x": round(x1, 1),
                "y": round(y1, 1),
                "width": round(x2 - x1, 1),
                "height": round(y2 - y1, 1),
            })

    counts: dict[str, int] = {}
    for d in detections:
        counts[d["class"]] = counts.get(d["class"], 0) + 1

    return {
        "ok": True,
        "detections": detections,
        "stats": {
            "total": len(detections),
            "counts_by_class": counts,
        },
        "image_size": {"width": w, "height": h},
        "timings_ms": {
            "decode": round(decode_ms, 2),
            "inference": round(infer_ms, 2),
            "total": round(decode_ms + infer_ms, 2),
        },
    }


@app.get("/")
def root():
    return {
        "service": "SCOS Snooker Ball Detector",
        "endpoints": ["/health", "/detect", "/detect/base64"],
        "model_repo": HF_MODEL_REPO,
    }
