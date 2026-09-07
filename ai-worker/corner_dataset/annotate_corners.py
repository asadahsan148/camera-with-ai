"""
Annotate snooker table corners in extracted frames.

Click the 4 corners in this order:
  1. Top-Left (TL)
  2. Top-Right (TR)
  3. Bottom-Right (BR)
  4. Bottom-Left (BL)

Controls:
  Left click  — place next corner
  Right click — reset current frame (start over)
  SPACE       — save annotation and go to next frame
  BACKSPACE   — go back to previous frame
  S           — skip frame (no annotation)
  Q / ESC     — quit (progress is saved automatically)

Output:
  labels/frame_0001.txt  — YOLO pose format: class x y (normalized)
  annotations.json       — all annotations in one file

Usage:
    python annotate_corners.py [--frames-dir frames] [--labels-dir labels]
"""

import argparse
import json
import os
import sys
import glob
import cv2
import numpy as np

CORNER_NAMES = ["TL", "TR", "BR", "BL"]
CORNER_COLORS = [
    (0, 255, 255),   # TL - yellow
    (255, 0, 255),   # TR - magenta
    (255, 255, 0),   # BR - cyan
    (0, 255, 0),     # BL - green
]
CORNER_RADIUS = 8
LINE_COLOR = (0, 200, 255)
LINE_THICKNESS = 2


class CornerAnnotator:
    def __init__(self, frames_dir, labels_dir, annotations_file="annotations.json"):
        self.frames_dir = frames_dir
        self.labels_dir = labels_dir
        self.annotations_file = annotations_file

        os.makedirs(labels_dir, exist_ok=True)

        self.frames = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
        if not self.frames:
            print(f"ERROR: No frames found in {frames_dir}")
            sys.exit(1)

        print(f"Found {len(self.frames)} frames to annotate")
        print()
        print("Instructions:")
        print("  Click 4 corners in order: TL -> TR -> BR -> BL")
        print("  SPACE  = save & next")
        print("  BACKSPACE = previous frame")
        print("  S      = skip (no annotation)")
        print("  R/right-click = reset current frame")
        print("  Q/ESC  = quit")
        print()

        self.current_idx = 0
        self.corners = []  # list of (x, y) for current frame
        self.annotations = self._load_existing()

        self.window_name = "SCOS Corner Annotator"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

    def _load_existing(self):
        if os.path.exists(self.annotations_file):
            with open(self.annotations_file, "r") as f:
                data = json.load(f)
                print(f"Loaded {len(data.get('frames', {}))} existing annotations")
                return data
        return {"frames": {}}

    def _save_annotations(self):
        with open(self.annotations_file, "w") as f:
            json.dump(self.annotations, f, indent=2)

    def _save_yolo_label(self, frame_path, corners, img_w, img_h):
        frame_name = os.path.splitext(os.path.basename(frame_path))[0]
        label_path = os.path.join(self.labels_dir, f"{frame_name}.txt")

        lines = []
        for i, (x, y) in enumerate(corners):
            nx = x / img_w
            ny = y / img_h
            lines.append(f"0 {nx:.6f} {ny:.6f}")

        with open(label_path, "w") as f:
            f.write("\n".join(lines) + "\n")

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.corners) < 4:
                self.corners.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.corners = []

    def _draw_overlay(self, img):
        display = img.copy()

        h, w = display.shape[:2]

        for i, (cx, cy) in enumerate(self.corners):
            color = CORNER_COLORS[i]
            cv2.circle(display, (cx, cy), CORNER_RADIUS, color, -1)
            cv2.circle(display, (cx, cy), CORNER_RADIUS + 3, color, 1)
            label = CORNER_NAMES[i]
            cv2.putText(display, label, (cx + 12, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if len(self.corners) >= 2:
            for i in range(len(self.corners) - 1):
                cv2.line(display, self.corners[i], self.corners[i + 1],
                         LINE_COLOR, LINE_THICKNESS)
        if len(self.corners) == 4:
            cv2.line(display, self.corners[3], self.corners[0],
                     LINE_COLOR, LINE_THICKNESS)
            cv2.line(display, self.corners[0], self.corners[2],
                     (100, 100, 100), 1)
            cv2.line(display, self.corners[1], self.corners[3],
                     (100, 100, 100), 1)

        info_lines = [
            f"Frame: {self.current_idx + 1}/{len(self.frames)}",
            f"File: {os.path.basename(self.frames[self.current_idx])}",
        ]
        if len(self.corners) < 4:
            info_lines.append(f"Click corner {len(self.corners) + 1}/4: {CORNER_NAMES[len(self.corners)]}")
        else:
            info_lines.append("All 4 corners placed. Press SPACE to save.")

        annotated_count = len(self.annotations.get("frames", {}))
        info_lines.append(f"Annotated total: {annotated_count}")

        y0 = 25
        for i, line in enumerate(info_lines):
            y = y0 + i * 25
            cv2.rectangle(display, (5, y - 18), (5 + 350, y + 5), (0, 0, 0), -1)
            cv2.putText(display, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return display

    def run(self):
        while 0 <= self.current_idx < len(self.frames):
            frame_path = self.frames[self.current_idx]
            img = cv2.imread(frame_path)
            if img is None:
                print(f"WARNING: Cannot read {frame_path}, skipping")
                self.current_idx += 1
                continue

            h, w = img.shape[:2]
            frame_name = os.path.basename(frame_path)
            frame_key = frame_name

            if frame_key in self.annotations.get("frames", {}) and not self.corners:
                existing = self.annotations["frames"][frame_key]
                self.corners = [(c["x"], c["y"]) for c in existing["corners"]]

            display = self._draw_overlay(img)

            cv2.imshow(self.window_name, display)

            key = cv2.waitKeyEx(1) & 0xFF

            if key == ord('q') or key == 27:
                print("\nQuitting. Annotations saved.")
                self._save_annotations()
                break
            elif key == ord(' '):
                if len(self.corners) == 4:
                    corner_data = [{"x": int(x), "y": int(y)} for x, y in self.corners]
                    self.annotations.setdefault("frames", {})[frame_key] = {
                        "file": frame_path,
                        "width": w,
                        "height": h,
                        "corners": corner_data,
                        "corner_names": CORNER_NAMES,
                    }
                    self._save_yolo_label(frame_path, self.corners, w, h)
                    self._save_annotations()
                    print(f"  Saved: {frame_name} ({len(self.annotations['frames'])} total)")
                    self.corners = []
                    self.current_idx += 1
                else:
                    print(f"  Need 4 corners, have {len(self.corners)}. Click more or press R to reset.")
            elif key == ord('s'):
                print(f"  Skipped: {frame_name}")
                self.corners = []
                self.current_idx += 1
            elif key == ord('r') or key == 8:
                if key == 8 and not self.corners:
                    if self.current_idx > 0:
                        self.current_idx -= 1
                        prev_name = os.path.basename(self.frames[self.current_idx])
                        if prev_name in self.annotations.get("frames", {}):
                            del self.annotations["frames"][prev_name]
                        print(f"  Back to: {prev_name}")
                else:
                    self.corners = []
                    print(f"  Reset corners for current frame")

        cv2.destroyAllWindows()
        total = len(self.annotations.get("frames", {}))
        print(f"\nAnnotation complete: {total} frames annotated")
        print(f"Labels saved to: {self.labels_dir}/")
        print(f"JSON saved to: {self.annotations_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate snooker table corners in frames")
    parser.add_argument("--frames-dir", default="frames", help="Directory with extracted frames")
    parser.add_argument("--labels-dir", default="labels", help="Output directory for YOLO labels")
    parser.add_argument("--annotations-file", default="annotations.json", help="JSON annotations file")
    args = parser.parse_args()

    annotator = CornerAnnotator(args.frames_dir, args.labels_dir, args.annotations_file)
    annotator.run()
