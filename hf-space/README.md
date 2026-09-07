---
title: SCOS Snooker Detector
emoji: 🎱
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
---

# SCOS Snooker Ball Detector — Hugging Face Space

This Space hosts a YOLOv8 model trained for snooker/pool ball detection.
It provides a FastAPI inference endpoint compatible with the SCOS backend `DetectorAdapter` interface.

## Setup

1. **Train the model** using `ai-worker/kaggle_train.ipynb` (Kaggle free GPU)
2. **Model weights** are pushed to Hugging Face Hub automatically by the notebook
3. **This Space** downloads the model on startup and serves inference via HTTP

## Configuration

Set these in **Space Settings → Variables and secrets**:

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_MODEL_REPO` | `your-username/scos-yolov8s` | HF Hub model repo ID |
| `MODEL_FILENAME` | `best.pt` | Model file to download |
| `CONFIDENCE_DEFAULT` | `0.25` | Default confidence threshold |
| `IOU_THRESHOLD` | `0.45` | NMS IoU threshold |
| `IMGSZ` | `640` | Inference image size |

## API

### `GET /health`
Returns model status and class names.

### `POST /detect`
Upload a JPEG image, get back SCOS-normalized detections.

```bash
curl -X POST https://yourname-scoker-detector.hf.space/detect \
  -F "file=@table_frame.jpg" \
  -F "confidence=0.4"
```

Response:
```json
{
  "ok": true,
  "detections": [
    { "class": "red", "confidence": 0.92, "x": 120, "y": 80, "width": 30, "height": 30 }
  ],
  "stats": { "total": 1, "counts_by_class": { "red": 1 } },
  "timings_ms": { "decode": 2.1, "inference": 45.3, "total": 47.4 }
}
```

### `POST /detect/base64`
Same as `/detect` but accepts base64-encoded image in JSON body.

## SCOS Integration

In the SCOS backend, use the `HFDetector` adapter:

```js
import { HFDetector } from './detectors/HFDetector.js';
const detector = new HFDetector({
  spaceUrl: 'https://yourname-scoker-detector.hf.space'
});
const result = await detector.detect(jpegBuffer, { confidence: 0.4 });
```
