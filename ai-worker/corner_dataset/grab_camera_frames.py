"""
Grab frames from all connected SCOS cameras.

Prerequisites:
  1. Backend running on http://localhost:5050
  2. NVR channels loaded (Cameras tab → Connect → Load channels)

Usage:
    python grab_camera_frames.py [--frames 20] [--interval 2] [--out frames]

Fetches the camera list (with RTSP URLs) from the backend API once,
then grabs frames directly from each RTSP stream using OpenCV.
Frames are saved as:
    frames/cam{N}_frame_{NN}.jpg

After capture, use annotate_corners.py to label the 4 table corners.
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error

import cv2

API = "http://localhost:5050"


def api_get(path):
    url = f"{API}{path}"
    with urllib.request.urlopen(url, timeout=15) as resp:
        return json.loads(resp.read())


def grab_frames_rtsp(rtsp_url, out_dir, cam_index, cam_name, num_frames, interval):
    """Grab frames directly from an RTSP URL using OpenCV."""
    fname_prefix = f"cam{cam_index}"
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open RTSP stream for {cam_name}")
        print(f"         URL: {rtsp_url[:60]}...")
        return 0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Stream opened: {width}x{height}")

    captured = 0
    for frame_num in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            # Try reconnect
            print(f"  [{frame_num+1}/{num_frames}] Frame grab failed, reconnecting...")
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            ret, frame = cap.read()
            if not ret:
                print(f"  [{frame_num+1}/{num_frames}] Reconnect failed, skipping")
                continue

        fname = f"{fname_prefix}_frame_{frame_num+1:02d}.jpg"
        fpath = os.path.join(out_dir, fname)
        cv2.imwrite(fpath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        captured += 1
        h, w = frame.shape[:2]
        fsize = os.path.getsize(fpath) // 1024
        print(f"  [{frame_num+1}/{num_frames}] {fname} ({w}x{h}, {fsize}KB)")

        if frame_num < num_frames - 1:
            time.sleep(interval)

    cap.release()
    return captured


def main():
    parser = argparse.ArgumentParser(description="Grab frames from all SCOS cameras")
    parser.add_argument("--frames", type=int, default=20, help="Frames per camera")
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between frames")
    parser.add_argument("--out", default="frames", help="Output directory")
    parser.add_argument(
        "--skip-keywords",
        default="Counter,Waiting,Parking,Channel",
        help="Comma-separated keywords to skip cameras by name (case-insensitive)",
    )
    args = parser.parse_args()
    skip_keywords = [k.strip().lower() for k in args.skip_keywords.split(",") if k.strip()]

    os.makedirs(args.out, exist_ok=True)

    # 1. Fetch camera list from backend (just once, to get RTSP URLs)
    print("Fetching camera list from backend...")
    try:
        data = api_get("/api/cameras")
    except urllib.error.URLError as e:
        print(f"ERROR: Cannot reach backend at {API}: {e}")
        sys.exit(1)

    cameras = data.get("cameras", [])
    all_connected = [c for c in cameras if c.get("connected") and c.get("rtspUrl")]

    # Filter out non-table cameras by name keywords
    connected = []
    for c in all_connected:
        name_lower = (c.get("name") or "").lower()
        if any(kw in name_lower for kw in skip_keywords):
            print(f"  SKIP: {c.get('name')} (matches skip keyword)")
            continue
        connected.append(c)

    if not connected:
        print("No table cameras found after filtering. Load NVR channels first via the UI.")
        sys.exit(1)

    print(f"Found {len(connected)} connected cameras:")
    for i, cam in enumerate(connected):
        print(f"  cam{i}: {cam.get('name', 'unnamed')} (ch={cam.get('channel')}, ip={cam.get('ip')})")

    # 2. Grab frames directly from each RTSP stream
    total_saved = 0
    for i, cam in enumerate(connected):
        cam_name = cam.get("name", f"cam{i}")
        rtsp_url = cam["rtspUrl"]
        print(f"\n--- Capturing {args.frames} frames from cam{i} ({cam_name}) ---")

        captured = grab_frames_rtsp(
            rtsp_url, args.out, i, cam_name, args.frames, args.interval
        )
        total_saved += captured
        print(f"  -> {captured}/{args.frames} frames captured for cam{i}")

    print(f"\n{'='*50}")
    print(f"Total frames saved: {total_saved}")
    print(f"Output directory: {os.path.abspath(args.out)}")
    print(f"\nNext step: python annotate_corners.py --frames-dir {args.out}")


if __name__ == "__main__":
    main()
