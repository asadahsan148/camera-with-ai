"""
Extract frames from VideoForAI.mp4 for training data.

Usage:
    python extract_frames.py [--fps 1] [--max-frames 200]

Extracts 1 frame per second by default. Adjust --fps for density.
Output: frames/frame_0001.jpg, frame_0002.jpg, ...
"""

import argparse
import os
import sys
import cv2


def extract_frames(video_path, output_dir, fps=1, max_frames=200):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        sys.exit(1)

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / video_fps if video_fps > 0 else 0

    print(f"Video: {video_path}")
    print(f"  FPS: {video_fps:.1f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {duration_sec:.1f}s")
    print(f"  Extracting 1 frame every {1/fps:.1f}s (fps={fps})")

    os.makedirs(output_dir, exist_ok=True)

    frame_interval = int(video_fps / fps) if video_fps > 0 else 1
    frame_interval = max(1, frame_interval)

    extracted = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            out_path = os.path.join(output_dir, f"frame_{extracted + 1:04d}.jpg")
            cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            extracted += 1
            print(f"  Extracted frame {extracted}: {out_path} ({frame.shape[1]}x{frame.shape[0]})")

            if extracted >= max_frames:
                print(f"  Reached max_frames limit ({max_frames})")
                break

        frame_idx += 1

    cap.release()
    print(f"\nDone: {extracted} frames extracted to {output_dir}/")
    return extracted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from video for training")
    parser.add_argument("--video", default="VideoForAI.mp4", help="Input video path")
    parser.add_argument("--output", default="frames", help="Output directory")
    parser.add_argument("--fps", type=float, default=1, help="Frames per second to extract")
    parser.add_argument("--max-frames", type=int, default=200, help="Maximum frames to extract")
    args = parser.parse_args()

    extract_frames(args.video, args.output, args.fps, args.max_frames)
