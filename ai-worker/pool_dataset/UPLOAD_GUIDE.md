# Upload Dataset to Hugging Face - Quick Guide

## Step 1: Install Requirements

```bash
pip install huggingface_hub
```

## Step 2: Login to Hugging Face

```bash
huggingface-cli login
```

You'll need a Hugging Face token:
1. Go to https://huggingface.co/settings/tokens
2. Create a new token with "write" permission
3. Copy and paste it when prompted

## Step 3: Configure the Upload Script

Edit `upload_to_huggingface.py`:
- Change `HF_USERNAME` to your Hugging Face username
- Optionally change `DATASET_NAME` to your preferred name

## Step 4: Run the Upload Script

```bash
python upload_to_huggingface.py
```

## Step 5: Use in Kaggle

Once uploaded, in your Kaggle notebook:

1. Click "Add Data" button
2. Select "Hugging Face" tab
3. Search for your dataset name
4. Click "Add"

The dataset will be mounted at `/kaggle/input/your-dataset-name/`

## Alternative: Direct Download in Kaggle

If you prefer, you can also download directly in your Kaggle notebook:

```python
from huggingface_hub import snapshot_download

dataset_path = snapshot_download(
    repo_id="YOUR_USERNAME/DATASET_NAME",
    repo_type="dataset",
    local_dir="./pool_dataset"
)

print(f"Dataset downloaded to: {dataset_path}")
```

## Verify Upload

After uploading, visit:
```
https://huggingface.co/datasets/YOUR_USERNAME/DATASET_NAME
```

You should see all your files (train, valid, test folders and data.yaml).
