---
title: SCOS Corner Detector
emoji: 🎱
colorFrom: green
colorTo: blue
sdk: gradio
app_port: 7860
pinned: false
license: apache-2.0
---

# SCOS Snooker Table Corner Detector

Detects the 4 corners of a snooker table for automatic AI-powered calibration.
Returns `TL, TR, BR, BL` keypoints — feeds directly into the SCOS perspective warp pipeline.

## API

### `POST /detect-corners`
Upload a JPEG frame, get back the 4 corner pixel coordinates.

```bash
curl -X POST https://asadahsan148-scos-corner-detector.hf.space/detect-corners \
  -F "file=@frame.jpg" \
  -F "confidence=0.4"
```

Response:
```json
{
  "ok": true,
  "found": true,
  "corners": [
    { "name": "TL", "x": 120, "y": 95,  "x_norm": 0.125, "y_norm": 0.088 },
    { "name": "TR", "x": 840, "y": 92,  "x_norm": 0.875, "y_norm": 0.085 },
    { "name": "BR", "x": 848, "y": 985, "x_norm": 0.883, "y_norm": 0.912 },
    { "name": "BL", "x": 112, "y": 988, "x_norm": 0.117, "y_norm": 0.915 }
  ],
  "confidence": 0.94,
  "image_size": { "width": 960, "height": 1080 },
  "timings_ms": { "inference": 48.2 }
}
```

A Gradio UI is also available at `/gradio` for interactive testing.

## Configuration

Set in **Space Settings → Variables**:

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_MODEL_REPO` | `asadahsan148/scos-corner-detector` | HF Hub model repo |
| `MODEL_FILENAME` | `best.pt` | Model weights file |
| `CONF_THRESHOLD` | `0.25` | Detection confidence threshold |
| `IMGSZ` | `640` | Inference image size |
