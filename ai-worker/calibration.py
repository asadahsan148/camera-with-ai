"""
SCOS Phase 1 — table calibration geometry.

Four admin-clicked corners of a playing surface -> perspective matrix ->
consistent top-down rectangle.

This module deliberately contains NO ball / colour / contour / tracking logic.
It is pure geometry so the vision baseline can be verified visually first.
"""

from __future__ import annotations

import time
from typing import Any, Sequence

import cv2
import numpy as np

CORNER_LABELS = ("top-left", "top-right", "bottom-right", "bottom-left")

# Snooker / English pool playing surface is 2:1 (e.g. 3569mm x 1778mm).
DEFAULT_ASPECT = 2.0
DEFAULT_SURFACE_WIDTH = 1200
# Extra band kept around the surface so cushions + pocket jaws stay visible.
# Expressed as a fraction of the surface HEIGHT so all four sides get the
# same physical width of table furniture.
DEFAULT_MARGIN = 0.10

MIN_SURFACE_WIDTH = 320
MAX_SURFACE_WIDTH = 2400
MAX_MARGIN = 0.35


def as_points(corners: Sequence[Any]) -> np.ndarray:
    """Validate and coerce a click list into a (4, 2) float array."""
    pts = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] != 4:
        raise ValueError("exactly 4 corners required, in order TL, TR, BR, BL")
    if not np.isfinite(pts).all():
        raise ValueError("corners contain non-finite values")
    return pts


def describe_quad(corners: Sequence[Any]) -> dict[str, Any]:
    """Geometry report for the clicked quad — used for calibration warnings."""
    p = as_points(corners)
    top = float(np.linalg.norm(p[1] - p[0]))
    right = float(np.linalg.norm(p[2] - p[1]))
    bottom = float(np.linalg.norm(p[2] - p[3]))
    left = float(np.linalg.norm(p[3] - p[0]))

    area = float(abs(cv2.contourArea(p.reshape(-1, 1, 2))))
    convex = bool(cv2.isContourConvex(p.reshape(-1, 1, 2).astype(np.int32)))

    mean_w = (top + bottom) / 2.0
    mean_h = (left + right) / 2.0
    aspect = float(mean_w / mean_h) if mean_h > 1e-6 else 0.0

    return {
        "edges": {
            "top": round(top, 2),
            "right": round(right, 2),
            "bottom": round(bottom, 2),
            "left": round(left, 2),
        },
        "area_px": round(area, 1),
        "convex": convex,
        "aspect": round(aspect, 3),
    }


def validate_corners(
    corners: Sequence[Any],
    frame_width: int | None = None,
    frame_height: int | None = None,
    target_aspect: float = DEFAULT_ASPECT,
) -> dict[str, Any]:
    """Return {ok, errors, warnings, quad} for a set of clicked corners."""
    errors: list[str] = []
    warnings: list[str] = []

    try:
        p = as_points(corners)
    except ValueError as exc:
        return {"ok": False, "errors": [str(exc)], "warnings": [], "quad": None}

    quad = describe_quad(p)

    if quad["area_px"] < 500:
        errors.append("Selected area is too small — click the four table corners wider apart.")
    if not quad["convex"]:
        errors.append(
            "Corners do not form a valid quadrilateral. "
            "Click in order: top-left, top-right, bottom-right, bottom-left."
        )

    edges = quad["edges"]
    if min(edges.values()) < 20:
        errors.append("Two corners are almost on top of each other.")

    if frame_width and frame_height:
        margin = 2.0
        outside = [
            CORNER_LABELS[i]
            for i, (x, y) in enumerate(p)
            if x < -margin or y < -margin or x > frame_width + margin or y > frame_height + margin
        ]
        if outside:
            errors.append(f"Corners outside the frame: {', '.join(outside)}")

    if not errors and target_aspect:
        drift = abs(quad["aspect"] - target_aspect) / target_aspect
        if drift > 0.6:
            warnings.append(
                f"Clicked quad looks {quad['aspect']:.2f}:1 but the target table is "
                f"{target_aspect:.2f}:1 — check the corner order / positions."
            )

    return {"ok": not errors, "errors": errors, "warnings": warnings, "quad": quad}


def build_output_spec(
    surface_width: int = DEFAULT_SURFACE_WIDTH,
    aspect: float = DEFAULT_ASPECT,
    margin: float = DEFAULT_MARGIN,
) -> dict[str, int | float]:
    """
    Work out the top-down canvas.

    The clicked playing surface is placed in the middle at `surface_width` x
    `surface_width / aspect`, with an equal physical band on every side so
    cushions and pocket jaws are not cropped away.
    """
    surface_width = int(max(MIN_SURFACE_WIDTH, min(MAX_SURFACE_WIDTH, int(surface_width))))
    aspect = float(aspect) if aspect and aspect > 0.05 else DEFAULT_ASPECT
    margin = float(max(0.0, min(MAX_MARGIN, margin)))

    surface_height = int(round(surface_width / aspect))
    band = int(round(margin * surface_height))

    return {
        "surface_width": surface_width,
        "surface_height": surface_height,
        "band": band,
        "margin": margin,
        "aspect": round(aspect, 4),
        "output_width": surface_width + 2 * band,
        "output_height": surface_height + 2 * band,
    }


def destination_corners(spec: dict[str, Any]) -> np.ndarray:
    """TL, TR, BR, BL of the playing surface inside the output canvas."""
    band = float(spec["band"])
    sw = float(spec["surface_width"])
    sh = float(spec["surface_height"])
    x0, y0 = band, band
    x1, y1 = band + sw - 1.0, band + sh - 1.0
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


def compute_matrix(
    corners: Sequence[Any],
    spec: dict[str, Any] | None = None,
    **spec_kwargs: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Perspective matrix mapping clicked corners onto the top-down canvas."""
    spec = spec or build_output_spec(**spec_kwargs)
    src = as_points(corners)
    dst = destination_corners(spec)
    matrix = cv2.getPerspectiveTransform(src, dst)
    return matrix, spec


def warp(
    image_bgr: np.ndarray,
    matrix: np.ndarray,
    output_width: int,
    output_height: int,
) -> tuple[np.ndarray, float]:
    """Apply the perspective matrix. Returns (top-down image, milliseconds)."""
    m = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    t0 = time.perf_counter()
    out = cv2.warpPerspective(
        image_bgr,
        m,
        (int(output_width), int(output_height)),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return out, (time.perf_counter() - t0) * 1000.0


def scale_corners(
    corners: Sequence[Any],
    from_size: tuple[int, int],
    to_size: tuple[int, int],
) -> np.ndarray:
    """Move corners between two resolutions of the same camera view."""
    p = as_points(corners)
    fw, fh = float(from_size[0]), float(from_size[1])
    tw, th = float(to_size[0]), float(to_size[1])
    if fw <= 0 or fh <= 0:
        return p
    return np.stack([p[:, 0] * (tw / fw), p[:, 1] * (th / fh)], axis=1).astype(np.float32)


def encode_jpeg(image_bgr: np.ndarray, quality: int = 82) -> bytes:
    ok, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


def draw_roi_overlay(
    image_bgr: np.ndarray,
    corners: Sequence[Any],
    *,
    label: str | None = None,
) -> np.ndarray:
    """Raw frame with the calibrated ROI drawn on it (diagnostics only)."""
    out = image_bgr.copy()
    p = as_points(corners).astype(np.int32)
    cv2.polylines(out, [p.reshape(-1, 1, 2)], True, (0, 220, 255), 2, cv2.LINE_AA)
    for i, (x, y) in enumerate(p):
        cv2.circle(out, (int(x), int(y)), 7, (0, 220, 255), -1, cv2.LINE_AA)
        cv2.putText(
            out,
            str(i + 1),
            (int(x) + 10, int(y) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )
    if label:
        cv2.putText(
            out, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2, cv2.LINE_AA
        )
    return out
