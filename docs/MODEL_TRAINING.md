# SCOS Model Training & Deployment Guide

This guide covers the full pipeline: **train** a snooker ball detection model for free, **store** it permanently, and **deploy** it as an accessible API.

## Architecture

```
Kaggle (free GPU)          Hugging Face Hub (free storage)      Hugging Face Space (free CPU)
┌────────────────┐         ┌──────────────────┐                ┌─────────────────────┐
│ kaggle_train   │         │ your-username/   │                │ FastAPI + YOLOv8    │
│ .ipynb         │──push──▶│ scos-yolov8s     │──download──────│ /detect endpoint    │
│                │         │ best.pt, onnx    │                │ yourname.hf.space   │
└────────────────┘         └──────────────────┘                └─────────┬───────────┘
                                                                         │
                                                               SCOS backend calls
                                                                         │
                                                               ┌─────────▼───────────┐
                                                               │ HFDetector.js       │
                                                               │ (DetectorAdapter)   │
                                                               └─────────────────────┘
```

## Step 1: Train on Kaggle (Free GPU — 30 hrs/week)

### Prerequisites
- Kaggle account (free): https://kaggle.com
- Hugging Face account (free): https://huggingface.co

### Instructions

1. **Upload dataset to Kaggle**:
   - Go to https://kaggle.com/datasets → New Dataset
   - Upload the `ai-worker/pool_dataset/` folder (zip it first if needed)
   - Name it `scos-pool-dataset` (or whatever you prefer)
   - Set it to Public or Private

2. **Create HF model repo**:
   - Go to https://huggingface.co/new → Model → Object Detection
   - Name it `scos-yolov8s` (or similar)
   - License: Apache 2.0

3. **Set up Kaggle Secrets**:
   - In your Kaggle notebook: Settings → Secrets → Add new secret
   - Name: `HF_TOKEN`
   - Value: Your HF token from https://huggingface.co/settings/tokens (write permission)

4. **Run the notebook**:
   - Open `ai-worker/kaggle_train.ipynb` in Kaggle
   - Set Accelerator to GPU (T4 x2 or P100)
   - Edit Cell 3: set `HF_REPO_ID = 'your-username/scos-yolov8s'`
   - Run all cells top to bottom
   - Training takes ~30-60 min for 100 epochs on ~800 images

5. **Model is now on HF Hub**:
   - Weights at `https://huggingface.co/your-username/scos-yolov8s`
   - Files: `best.pt`, `best.onnx`, `results.png`, `README.md`

## Step 2: Deploy to Hugging Face Space (Free CPU hosting)

### Instructions

1. **Create a new Space**:
   - Go to https://huggingface.co/new-space
   - SDK: **Docker**
   - Hardware: **CPU basic** (free)
   - Name: `scos-detector`

2. **Upload Space files**:
   - Copy the contents of the `hf-space/` directory to the Space repo
   - Files needed: `app.py`, `requirements.txt`, `Dockerfile`, `README.md`

3. **Configure Space variables**:
   - Settings → Variables and secrets
   - Add: `HF_MODEL_REPO` = `your-username/scos-yolov8s`
   - Add: `MODEL_FILENAME` = `best.pt`

4. **Space builds and starts**:
   - First build takes ~5 min (installs ultralytics + downloads model)
   - Health check: `https://yourname-scos-detector.hf.space/health`
   - Test detection:
     ```bash
     curl -X POST https://yourname-scos-detector.hf.space/detect \
       -F "file=@test_image.jpg" \
       -F "confidence=0.4"
     ```

## Step 3: Connect SCOS Backend

### Option A: Use HF Space (remote inference)

1. Edit `backend/.env`:
   ```
   HF_SPACE_URL=https://yourname-scos-detector.hf.space
   ```

2. Restart backend. The `hf-space` detector is now available in the Model Test tab.

### Option B: Local inference (production — zero latency)

1. Download model from HF Hub:
   ```bash
   huggingface-cli download your-username/scos-yolov8s best.pt --local-dir ai-worker/models/
   ```

2. Add a local YOLO inference endpoint to `ai-worker/app.py` (future task)

3. Point `HF_SPACE_URL` to `http://127.0.0.1:5051` (local AI worker)

## File Overview

| File | Purpose |
|------|---------|
| `ai-worker/kaggle_train.ipynb` | Kaggle notebook: train YOLOv8, export ONNX, push to HF Hub |
| `hf-space/app.py` | FastAPI server: loads model from HF Hub, serves /detect |
| `hf-space/requirements.txt` | Python deps for the Space |
| `hf-space/Dockerfile` | Docker image for HF Space |
| `hf-space/README.md` | Space README with YAML frontmatter |
| `backend/src/detectors/HFDetector.js` | SCOS DetectorAdapter for HF Space |
| `backend/src/detectors/index.js` | Updated registry (now includes `hf-space`) |
| `backend/.env.example` | Updated with `HF_SPACE_URL` variable |

## Cost Summary

| Platform | What's Free | Limits |
|----------|-------------|--------|
| **Kaggle** | GPU (P100/T4), 30 hrs/week | 9-12 hr sessions, 20 GB disk |
| **HF Hub** | Model storage, unlimited repos | Public repos free; private limited |
| **HF Space (CPU)** | Hosting, 2 vCPU, 16 GB RAM | Auto-sleep after 48h inactivity |
| **HF Space (ZeroGPU)** | 5 min/day GPU | Too limited for production |

**Total cost: $0** for training, storage, and deployment.
