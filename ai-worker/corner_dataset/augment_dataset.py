"""
Generative augmentation for snooker table corner dataset.

Reads annotated frames + corner labels from annotations.json,
applies multiple augmentation variants per frame using Albumentations,
and saves augmented images + YOLO pose labels.

Usage:
    python augment_dataset.py [--variants 10] [--out augmented]

With 107 annotated frames x 10 variants = 1,070+ augmented frames.
"""

import argparse
import json
import os
import sys
import random

import cv2
import numpy as np
import albumentations as A

CORNER_NAMES = ["TL", "TR", "BR", "BL"]


def build_pipeline():
    """Build a diverse augmentation pipeline that supports keypoints."""
    return A.Compose(
        [
            # Geometric — small rotations and perspective warps simulate camera angle variation
            A.OneOf(
                [
                    A.Rotate(limit=15, p=1.0, border_mode=cv2.BORDER_REFLECT_101),
                    A.Affine(
                        translate_percent=0.05, scale=0.1, rotate=10, p=1.0,
                        border_mode=cv2.BORDER_REFLECT_101,
                    ),
                    A.Perspective(scale=(0.05, 0.1), p=1.0),
                ],
                p=0.8,
            ),
            # Photometric — simulate different lighting conditions
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.3, contrast_limit=0.3, p=1.0
                    ),
                    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
                    A.HueSaturationValue(
                        hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=20, p=1.0
                    ),
                    A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                ],
                p=0.9,
            ),
            # Noise / blur — simulate camera quality variation
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MotionBlur(blur_limit=7, p=1.0),
                    A.GaussNoise(std_range=(0.02, 0.1), p=1.0),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                ],
                p=0.4,
            ),
            # Occasional coarse dropout — simulate partial occlusion
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(8, 30),
                hole_width_range=(8, 30),
                fill="random",
                p=0.15,
            ),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=True),
    )


def build_flip_pipeline():
    """Horizontal flip pipeline — keypoints need reordering after flip."""
    return A.Compose(
        [A.HorizontalFlip(p=1.0)],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=True),
    )


def save_yolo_label(label_path, corners, img_w, img_h):
    """Save corners in YOLO pose format: class x y (normalized)."""
    lines = []
    for x, y in corners:
        nx = x / img_w
        ny = y / img_h
        lines.append(f"0 {nx:.6f} {ny:.6f}")
    with open(label_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Augment snooker corner dataset")
    parser.add_argument("--variants", type=int, default=10, help="Augmented variants per frame")
    parser.add_argument("--out", default="augmented", help="Output directory")
    parser.add_argument(
        "--annotations", default="annotations.json", help="Annotations JSON file"
    )
    parser.add_argument("--frames-dir", default="frames", help="Directory with original frames")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load annotations
    if not os.path.exists(args.annotations):
        print(f"ERROR: {args.annotations} not found")
        sys.exit(1)

    with open(args.annotations) as f:
        ann_data = json.load(f)

    annotated_frames = ann_data.get("frames", {})
    if not annotated_frames:
        print("ERROR: No annotated frames found in annotations.json")
        sys.exit(1)

    print(f"Loaded {len(annotated_frames)} annotated frames")

    # Also include old frames that have labels
    import glob

    old_label_dir = "labels"
    old_frames = []
    for label_file in glob.glob(os.path.join(old_label_dir, "*.txt")):
        frame_name = os.path.splitext(os.path.basename(label_file))[0]
        frame_path = os.path.join(args.frames_dir, f"{frame_name}.jpg")
        if not os.path.exists(frame_path):
            continue
        if frame_name in annotated_frames:
            continue  # Already in annotations
        # Read label file
        with open(label_file) as f:
            lines = f.readlines()
        corners = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 3:
                nx, ny = float(parts[1]), float(parts[2])
                corners.append((nx, ny))  # normalized
        if len(corners) == 4:
            old_frames.append((frame_name, frame_path, corners, True))  # True = normalized

    # Build frame list: (frame_name, frame_path, corners, is_normalized)
    frames_to_augment = []
    for frame_name, info in annotated_frames.items():
        frame_path = info["file"]
        if not os.path.exists(frame_path):
            # Try in frames_dir
            frame_path = os.path.join(args.frames_dir, frame_name)
        if not os.path.exists(frame_path):
            print(f"  WARNING: Skipping {frame_name} — file not found")
            continue
        corners = [(c["x"], c["y"]) for c in info["corners"]]
        frames_to_augment.append((frame_name, frame_path, corners, False))

    # Add old frames with normalized coords
    for frame_name, frame_path, corners, is_norm in old_frames:
        frames_to_augment.append((frame_name, frame_path, corners, is_norm))

    print(f"Total frames to augment: {len(frames_to_augment)}")
    print(f"Variants per frame: {args.variants}")
    print(f"Expected output: ~{len(frames_to_augment) * args.variants} augmented frames")
    print()

    # Create output directories
    img_out = os.path.join(args.out, "images")
    lbl_out = os.path.join(args.out, "labels")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    pipeline = build_pipeline()
    flip_pipeline = build_flip_pipeline()

    total_generated = 0
    total_skipped = 0

    for frame_name, frame_path, corners, is_normalized in frames_to_augment:
        img = cv2.imread(frame_path)
        if img is None:
            print(f"  WARNING: Cannot read {frame_path}, skipping")
            continue

        h, w = img.shape[:2]

        # Convert normalized corners to pixel coords if needed
        if is_normalized:
            pixel_corners = [(x * w, y * h) for x, y in corners]
        else:
            pixel_corners = [(float(x), float(y)) for x, y in corners]

        base_name = os.path.splitext(frame_name)[0]

        for variant_idx in range(args.variants):
            try:
                # Every 3rd variant, apply horizontal flip first
                if variant_idx % 3 == 2:
                    result = flip_pipeline(image=img, keypoints=pixel_corners)
                    flipped_img = result["image"]
                    flipped_kp = result["keypoints"]
                    if len(flipped_kp) != 4:
                        total_skipped += 1
                        continue
                    # Reorder corners after flip: TL<->TR, BL<->BR
                    reordered = [flipped_kp[1], flipped_kp[0], flipped_kp[3], flipped_kp[2]]
                    # Then apply main pipeline
                    result = pipeline(image=flipped_img, keypoints=reordered)
                else:
                    result = pipeline(image=img, keypoints=pixel_corners)

                aug_img = result["image"]
                aug_kp = result["keypoints"]

                if len(aug_kp) != 4:
                    total_skipped += 1
                    continue

                # Save image
                out_name = f"{base_name}_aug{variant_idx:02d}.jpg"
                out_img_path = os.path.join(img_out, out_name)
                cv2.imwrite(out_img_path, aug_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

                # Save label
                out_lbl_path = os.path.join(lbl_out, f"{base_name}_aug{variant_idx:02d}.txt")
                aug_h, aug_w = aug_img.shape[:2]
                save_yolo_label(out_lbl_path, aug_kp, aug_w, aug_h)

                total_generated += 1

            except Exception as e:
                print(f"  WARNING: Augmentation failed for {frame_name} variant {variant_idx}: {e}")
                total_skipped += 1

        # Progress
        if (total_generated + total_skipped) % 50 == 0:
            print(f"  Progress: {total_generated} generated, {total_skipped} skipped")

    print(f"\n{'='*50}")
    print(f"Total augmented frames generated: {total_generated}")
    print(f"Total skipped: {total_skipped}")
    print(f"Output images: {os.path.abspath(img_out)}")
    print(f"Output labels: {os.path.abspath(lbl_out)}")
    print(f"\nNext steps:")
    print(f"  1. Copy original + augmented frames into Kaggle dataset")
    print(f"  2. Upload to Kaggle and retrain with the larger dataset")


if __name__ == "__main__":
    main()
