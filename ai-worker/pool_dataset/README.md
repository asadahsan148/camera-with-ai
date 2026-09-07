---
license: mit
task_categories:
- object-detection
tags:
- yolo
- computer-vision
- pool
- billiards
size_categories:
- n<1K
---

# Pool Detection Dataset

This is a YOLO format dataset for object detection tasks.

## Dataset Structure

```
pool/
  train/
    images/
    labels/
  valid/
    images/
    labels/
  test/
    images/
    labels/
  data.yaml
```

## Classes

The dataset contains 16 classes representing pool balls (numbered 0-15).

## Usage

### In Python

```python
from datasets import load_dataset

dataset = load_dataset("nhantu2107/pool-detection")
```

### In Kaggle

1. Go to your Kaggle notebook
2. Click "Add data" -> "Hugging Face"
3. Search for "pool-detection"
4. Click "Add"

### With YOLOv8/YOLOv11

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(data='data.yaml', epochs=100)
```

## Citation

If you use this dataset, please cite appropriately.
