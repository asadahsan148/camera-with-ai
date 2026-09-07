# SCOS Auto-Calibration — AI Corner Detection

Replaces manual click-to-calibrate with a YOLOv8-pose model that automatically
detects the 4 snooker table corners (TL → TR → BR → BL) from any camera frame.

## Architecture

```
Camera (RTSP)
     │
     ▼
FrameSource (grab 1 JPEG)
     │
     ▼
POST /api/auto-calibrate/:cameraId
     │
     ├──▶ HF Space /detect-corners   ← YOLOv8-pose model
     │         returns: TL, TR, BR, BL pixel coords
     │
     └──▶ AI Worker /calibration/matrix
               computes perspective transform matrix
               (same as manual calibration path)
               │
               ▼
         TableStore.createOrUpdateTable()
         saves calibration to backend/data/tables.json
```

## Step-by-Step: Build the Auto-Calibration Model

### Step 1 — Annotate frames (your machine, ~30 min)

100 frames have been extracted from `VideoForAI.mp4` to `ai-worker/corner_dataset/frames/`.

Run the annotation tool:
```bash
cd ai-worker/corner_dataset
python annotate_corners.py
```

Click 4 corners per frame in order: **TL → TR → BR → BL**, press SPACE to save.
Aim for **at least 50 annotated frames** (more = better accuracy).

Labels are saved to `ai-worker/corner_dataset/labels/`.

### Step 2 — Upload dataset to Kaggle

1. Zip `ai-worker/corner_dataset/` (only `frames/` + `labels/` needed)
2. Go to https://kaggle.com/datasets → New Dataset
3. Name it `scos-corners`, upload zip
4. Note the dataset path (will be `/kaggle/input/scos-corners`)

### Step 3 — Train on Kaggle (free GPU)

1. Open `ai-worker/corner_dataset/kaggle_train_corners.ipynb` in Kaggle
2. Set Accelerator → **GPU (P100 or T4)**
3. Add dataset: Add Data → Your Datasets → `scos-corners`
4. Add Secret: `HF_TOKEN` = your HF write token
5. Edit Cell 3: set `HF_REPO_ID = 'your-username/scos-corner-detector'`
6. Run all cells (~20-40 min for 150 epochs on ~50 frames)

Model is pushed to HF Hub automatically.

### Step 4 — Deploy HF Space

1. Create new Space: https://huggingface.co/new-space
   - SDK: **Docker**, Hardware: **CPU basic** (free)
   - Name: `scos-corners`
2. Upload files from `hf-space-corners/`:
   - `app.py`, `requirements.txt`, `Dockerfile`, `README.md`
3. Space Settings → Variables:
   - `HF_MODEL_REPO` = `your-username/scos-corner-detector`
4. Space URL: `https://your-username-scos-corners.hf.space`

Test it:
```bash
curl https://your-username-scos-corners.hf.space/health
```

### Step 5 — Connect to SCOS backend

Add to `backend/.env`:
```
HF_CORNER_SPACE_URL=https://your-username-scos-corners.hf.space
```

Restart the backend. Auto-calibration is now live.

## API Endpoints

### `GET /api/auto-calibrate/status`
Check if the HF Space is reachable.
```json
{ "hf_space_configured": true, "hf_space_reachable": true }
```

### `POST /api/auto-calibrate/:cameraId`
Grab a frame and detect corners (preview only, does not save).
```json
{
  "corners": [{"x": 120, "y": 95}, {"x": 840, "y": 92}, ...],
  "confidence": 0.94,
  "jpeg_b64": "..."
}
```

### `POST /api/auto-calibrate/:cameraId/save`
Detect corners AND save as calibration to `tables.json`.
```json
{ "ok": true, "table": {...}, "auto_calibrated": true }
```

## File Overview

| File | Purpose |
|------|---------|
| `ai-worker/corner_dataset/VideoForAI.mp4` | Training video |
| `ai-worker/corner_dataset/extract_frames.py` | Extract JPEGs from video |
| `ai-worker/corner_dataset/annotate_corners.py` | Click-to-annotate tool |
| `ai-worker/corner_dataset/labels/*.txt` | YOLO point labels (4 per frame) |
| `ai-worker/corner_dataset/annotations.json` | Annotations backup |
| `ai-worker/corner_dataset/kaggle_train_corners.ipynb` | Kaggle training notebook |
| `hf-space-corners/app.py` | HF Space FastAPI server |
| `hf-space-corners/Dockerfile` | Docker image |
| `backend/src/autoCalibrationRoutes.js` | Backend auto-cal API |

## What's Next

Once the model is deployed and cameras are connected:

1. **Backend** grabs a live frame from the camera
2. **HF Space** detects the 4 corners automatically
3. **AI Worker** computes the perspective matrix
4. **TableStore** saves the calibration
5. **No human click required** — full auto-calibration in ~1 second

When more cameras are connected, just call `POST /api/auto-calibrate/:cameraId/save`
for each one. Done.
