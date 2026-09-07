"""
Snooker AI worker.

Two independent surfaces:
  /calibration/*  — Phase 1 table geometry (perspective correction only)
  /analyze        — legacy HSV ball detector (kept for debug, not the baseline)
"""

from __future__ import annotations

import base64
import json
import time
from typing import Any, Dict

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import calibration as calib
from detector import FrameStateMachine

app = FastAPI(title="Snooker AI Worker", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions: Dict[str, FrameStateMachine] = {}


@app.get("/health")
def health():
    return {
        "ok": True,
        "sessions": list(sessions.keys()),
        "features": {
            "calibration": True,
            "legacy_detector": True,
            "draw_boxes": True,
        },
    }


# ---------------------------------------------------------------------------
# Phase 1 — table calibration (geometry only, no ball detection)
# ---------------------------------------------------------------------------


def _parse_corners(raw: Any) -> list[list[float]]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise HTTPException(400, f"corners is not valid JSON: {exc}") from exc
    try:
        return calib.as_points(raw).tolist()
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/calibration/matrix")
def calibration_matrix(payload: Dict[str, Any]):
    """Validate four clicked corners and return the perspective matrix."""
    corners = _parse_corners(payload.get("corners"))
    spec = calib.build_output_spec(
        surface_width=int(payload.get("surface_width") or calib.DEFAULT_SURFACE_WIDTH),
        aspect=float(payload.get("aspect") or calib.DEFAULT_ASPECT),
        margin=float(
            payload.get("margin") if payload.get("margin") is not None else calib.DEFAULT_MARGIN
        ),
    )
    check = calib.validate_corners(
        corners,
        frame_width=payload.get("frame_width"),
        frame_height=payload.get("frame_height"),
        target_aspect=spec["aspect"],
    )
    if not check["ok"]:
        return {"ok": False, "errors": check["errors"], "warnings": check["warnings"]}

    matrix, spec = calib.compute_matrix(corners, spec)
    return {
        "ok": True,
        "matrix": matrix.tolist(),
        "output": spec,
        "quad": check["quad"],
        "warnings": check["warnings"],
        "corner_labels": list(calib.CORNER_LABELS),
    }


@app.post("/calibration/warp")
async def calibration_warp(
    file: UploadFile = File(...),
    corners: str = Form(...),
    surface_width: int = Form(calib.DEFAULT_SURFACE_WIDTH),
    aspect: float = Form(calib.DEFAULT_ASPECT),
    margin: float = Form(calib.DEFAULT_MARGIN),
    draw_overlay: bool = Form(False),
    table_label: str = Form(""),
):
    """Perspective-correct one frame. Returns the top-down JPEG + timings."""
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty image")

    t_decode = time.perf_counter()
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "Could not decode image")
    decode_ms = (time.perf_counter() - t_decode) * 1000.0
    h, w = image.shape[:2]

    pts = _parse_corners(corners)
    spec = calib.build_output_spec(surface_width=surface_width, aspect=aspect, margin=margin)
    check = calib.validate_corners(pts, w, h, target_aspect=spec["aspect"])
    if not check["ok"]:
        raise HTTPException(400, "; ".join(check["errors"]))

    matrix, spec = calib.compute_matrix(pts, spec)
    warped, transform_ms = calib.warp(image, matrix, spec["output_width"], spec["output_height"])

    t_encode = time.perf_counter()
    warped_jpeg = calib.encode_jpeg(warped)
    payload: Dict[str, Any] = {
        "ok": True,
        "matrix": matrix.tolist(),
        "output": spec,
        "quad": check["quad"],
        "warnings": check["warnings"],
        "source_size": {"w": w, "h": h},
        "warped_jpeg_base64": base64.b64encode(warped_jpeg).decode("ascii"),
    }
    if draw_overlay:
        overlay = calib.draw_roi_overlay(image, pts, label=table_label or None)
        payload["overlay_jpeg_base64"] = base64.b64encode(calib.encode_jpeg(overlay)).decode(
            "ascii"
        )
    encode_ms = (time.perf_counter() - t_encode) * 1000.0

    payload["timings_ms"] = {
        "decode": round(decode_ms, 2),
        "transform": round(transform_ms, 3),
        "encode": round(encode_ms, 2),
        "total": round(decode_ms + transform_ms + encode_ms, 2),
    }
    return payload


# ---------------------------------------------------------------------------
# Phase 2B — draw pretrained detections (no detection logic here)
# ---------------------------------------------------------------------------

_BOX_COLORS = {
    "red": (60, 60, 230),
    "yellow": (0, 220, 240),
    "green": (60, 180, 60),
    "brown": (40, 80, 140),
    "blue": (220, 120, 40),
    "pink": (180, 80, 255),
    "black": (20, 20, 20),
    "cue_ball": (240, 240, 240),
}


@app.post("/vision/draw_boxes")
async def draw_boxes(
    file: UploadFile = File(...),
    detections: str = Form("[]"),
    confidence_threshold: float = Form(0.0),
):
    """Overlay SCOS-normalized boxes on a warped table JPEG."""
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty image")
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "Could not decode image")

    try:
        boxes = json.loads(detections) if isinstance(detections, str) else detections
    except json.JSONDecodeError as exc:
        raise HTTPException(400, f"detections JSON invalid: {exc}") from exc
    if not isinstance(boxes, list):
        raise HTTPException(400, "detections must be a list")

    thr = float(confidence_threshold or 0.0)
    out = image.copy()
    drawn = 0
    for b in boxes:
        conf = float(b.get("confidence") or 0)
        if conf < thr:
            continue
        cls = str(b.get("class") or "unknown")
        x = int(round(float(b.get("x") or 0)))
        y = int(round(float(b.get("y") or 0)))
        w = int(round(float(b.get("width") or 0)))
        h = int(round(float(b.get("height") or 0)))
        if w < 2 or h < 2:
            continue
        color = _BOX_COLORS.get(cls, (0, 220, 255))
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)
        label = f"{cls} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        y0 = max(0, y - th - 6)
        cv2.rectangle(out, (x, y0), (x + tw + 6, y), color, -1)
        cv2.putText(
            out,
            label,
            (x + 3, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0) if cls != "black" else (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        drawn += 1

    return {
        "ok": True,
        "drawn": drawn,
        "image_jpeg_base64": base64.b64encode(calib.encode_jpeg(out, quality=85)).decode("ascii"),
    }


@app.post("/sessions/{session_id}/reset")
def reset_session(session_id: str):
    sessions[session_id] = FrameStateMachine()
    return {"ok": True, "state": "IDLE"}


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    sessions.pop(session_id, None)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Legacy HSV/contour ball detector — debug only, not part of the Phase 1 baseline
# ---------------------------------------------------------------------------


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    session_id: str = Form("default"),
):
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty image")

    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "Could not decode image")

    if session_id not in sessions:
        sessions[session_id] = FrameStateMachine()

    result = sessions[session_id].update(image)
    return {
        "session_id": session_id,
        "state": result["state"],
        "events": result["events"],
        "analysis": result["analysis"],
    }


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    sm = sessions.get(session_id)
    if not sm:
        return {"session_id": session_id, "state": "NONE"}
    return {"session_id": session_id, "state": sm.state, "frame_id": sm.frame_id}
