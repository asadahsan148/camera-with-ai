"""
SCOS Table Corner Detector — Hugging Face Space (Gradio SDK)

Runs a YOLOv8-pose model that returns the 4 snooker table corner keypoints
(TL, TR, BR, BL) for automatic calibration.

Uses FastAPI for the /detect-corners and /health endpoints (same API as before)
and mounts Gradio at /gradio for a visual UI.
"""

from __future__ import annotations

import base64, os, time
from typing import Any

import cv2
import numpy as np
import gradio as gr
from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

HF_MODEL_REPO  = os.environ.get("HF_MODEL_REPO",  "asadahsan148/scos-corner-detector")
MODEL_FILENAME = os.environ.get("MODEL_FILENAME",  "best.pt")
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.25"))
IMGSZ          = int(os.environ.get("IMGSZ", "640"))

CORNER_NAMES = ["TL", "TR", "BR", "BL"]
CORNER_COLORS = [(0,255,255),(255,0,255),(255,255,0),(0,255,0)]

_model: YOLO | None = None

def get_model() -> YOLO:
    global _model
    if _model is not None:
        return _model
    t0 = time.perf_counter()
    path = hf_hub_download(repo_id=HF_MODEL_REPO, filename=MODEL_FILENAME)
    _model = YOLO(path)
    print(f"[scos-corners] Model loaded in {(time.perf_counter()-t0)*1000:.0f}ms")
    return _model

def _run_inference(img: np.ndarray, conf: float) -> dict:
    h, w = img.shape[:2]
    model = get_model()
    t0 = time.perf_counter()
    results = model(img, conf=conf, imgsz=IMGSZ, verbose=False)
    infer_ms = (time.perf_counter() - t0) * 1000
    r = results[0]

    if r.keypoints is None or len(r.keypoints.xy) == 0:
        return {"ok": True, "found": False, "corners": None,
                "confidence": None, "image_size": {"width": w, "height": h},
                "timings_ms": {"inference": round(infer_ms, 2)}}

    best_idx = int(r.boxes.conf.argmax())
    box_conf = float(r.boxes.conf[best_idx])
    kps_xy = r.keypoints.xy[best_idx].cpu().numpy()
    kps_conf = r.keypoints.conf[best_idx].cpu().numpy() if r.keypoints.conf is not None else None

    corners = []
    for i, (x, y) in enumerate(kps_xy):
        pt = {"name": CORNER_NAMES[i], "x": round(float(x), 2), "y": round(float(y), 2),
              "x_norm": round(float(x)/w, 6), "y_norm": round(float(y)/h, 6)}
        if kps_conf is not None:
            pt["confidence"] = round(float(kps_conf[i]), 4)
        corners.append(pt)

    return {"ok": True, "found": True, "corners": corners,
            "confidence": round(box_conf, 4),
            "image_size": {"width": w, "height": h},
            "timings_ms": {"inference": round(infer_ms, 2)}}

# ─── FastAPI app for API endpoints ───────────────────────────────────

app = FastAPI(title="SCOS Corner Detector", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

@app.get("/health")
def health():
    return {"ok": True, "model_repo": HF_MODEL_REPO,
            "model_loaded": _model is not None, "corner_names": CORNER_NAMES}

@app.post("/detect-corners")
async def detect_corners(
    file: UploadFile = File(...),
    confidence: float = Form(CONF_THRESHOLD),
):
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty image")
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode image")
    return _run_inference(img, confidence)

@app.post("/detect-corners/base64")
async def detect_corners_base64(payload: dict[str, Any]):
    image_data = payload.get("image", "")
    if not image_data:
        raise HTTPException(400, "image field required")
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]
    try:
        raw = base64.b64decode(image_data)
    except Exception:
        raise HTTPException(400, "Invalid base64")
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode image")
    confidence = float(payload.get("confidence", CONF_THRESHOLD))
    return _run_inference(img, confidence)

@app.get("/")
def root():
    return {"service": "SCOS Table Corner Detector",
            "endpoints": ["/health", "/detect-corners", "/detect-corners/base64", "/gradio"],
            "model_repo": HF_MODEL_REPO, "corners": CORNER_NAMES}

# ─── Gradio UI mounted at /gradio ────────────────────────────────────

def detect_fn(image, confidence):
    if image is None:
        return None, '{"ok": false, "error": "No image provided"}'
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    result = _run_inference(img, confidence)
    if not result["found"]:
        return image, str(result)
    annotated = image.copy()
    kps = [(c["x"], c["y"]) for c in result["corners"]]
    for i, (x, y) in enumerate(kps):
        xi, yi = int(x), int(y)
        cv2.circle(annotated, (xi, yi), 10, CORNER_COLORS[i], -1)
        cv2.putText(annotated, CORNER_NAMES[i], (xi+12, yi-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, CORNER_COLORS[i], 2)
    pts = np.array(kps, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(annotated, [pts], isClosed=True, color=(0, 200, 255), thickness=2)
    return annotated, str(result)

with gr.Blocks(title="SCOS Corner Detector") as demo:
    gr.Markdown("# SCOS Snooker Table Corner Detector")
    gr.Markdown("Upload a frame to detect 4 table corners (TL, TR, BR, BL).")
    with gr.Row():
        inp_img = gr.Image(label="Input Frame", type="numpy")
        out_img = gr.Image(label="Annotated Result")
    conf_slider = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Confidence")
    out_json = gr.Textbox(label="JSON Result", lines=8)
    btn = gr.Button("Detect Corners")
    btn.click(fn=detect_fn, inputs=[inp_img, conf_slider], outputs=[out_img, out_json])

app = gr.mount_gradio_app(app, demo, path="/gradio")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
