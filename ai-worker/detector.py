"""
Snooker detector pipeline (club CCTV):

Architecture (best-case live loop):
  WATCHING  → cheap motion only
  ACTIVE    → something moved → ball detector + track
  SETTLING  → balls stopped → evaluate events → WATCHING

Per detect sample:
  1. Closest green table quadrilateral
  2. Crop + perspective warp + light normalize
  3. Balls strictly inside one table border
  4. Centroid track / stop check
  5. Events only when settled (RACK / START / END)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

MAX_REDS = 15
MAX_BALLS = 22


@dataclass
class BallCounts:
    red: int = 0
    white: int = 0
    yellow: int = 0
    green: int = 0
    brown: int = 0
    blue: int = 0
    pink: int = 0
    black: int = 0

    def total(self) -> int:
        return (
            self.red
            + self.white
            + self.yellow
            + self.green
            + self.brown
            + self.blue
            + self.pink
            + self.black
        )


def _preprocess_cctv(image_bgr: np.ndarray, quality: dict[str, Any] | None = None) -> np.ndarray:
    """
    Stabilize bad club CCTV only when needed (dark / blurry / tiny).
    Good frames stay nearly untouched so counts don't inflate.
    """
    q = quality or _frame_quality(image_bgr)
    img = image_bgr
    if q.get("blurry") or q.get("low_res") or q.get("dark"):
        img = cv2.bilateralFilter(img, d=5, sigmaColor=35, sigmaSpace=35)

    mean = float(img.mean())
    if q.get("dark") or mean < 55:
        gain = min(2.0, 85.0 / max(mean, 1.0))
        img = np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)

    if q.get("dark") or q.get("blurry"):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
        l = clahe.apply(l)
        img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    return img


def _frame_quality(image_bgr: np.ndarray) -> dict[str, Any]:
    h, w = image_bgr.shape[:2]
    mean = float(image_bgr.mean())
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # High Laplacian variance ≈ sharp; very low ≈ blur / heavy compress
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return {
        "mean": mean,
        "sharp": sharp,
        "low_res": min(h, w) < 400,
        "dark": mean < 60,
        "blurry": sharp < 35,
        "poor": mean < 60 or sharp < 35,  # low_res alone → try strict first, then fallback
    }


def _cloth_mask(image_bgr: np.ndarray, *, for_roi: bool = False, wide: bool = False) -> np.ndarray:
    """
    Table cloth HSV mask.

    for_roi=True: heavy close to get a solid table region.
    for_roi=False: light morph so ball-sized holes stay open (critical —
    a 13x13 close was absorbing reds into the cloth and under-counting).
    wide=True: darker / cast CCTV (used in relaxed pass).
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    if wide or for_roi:
        lo_s, lo_v = 25, 20
        h0, h1 = 28, 100
    else:
        lo_s, lo_v = 45, 35
        h0, h1 = 35, 95
    cloth = cv2.inRange(hsv, np.array([h0, lo_s, lo_v]), np.array([h1, 255, 255]))
    cloth = cv2.bitwise_or(
        cloth, cv2.inRange(hsv, np.array([h1, lo_s, lo_v]), np.array([120, 255, 255]))
    )
    if for_roi:
        cloth = cv2.morphologyEx(cloth, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        cloth = cv2.morphologyEx(cloth, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    else:
        cloth = cv2.morphologyEx(cloth, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cloth = cv2.morphologyEx(cloth, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return cloth


def _classify_mean_bgr(mean_bgr: tuple[float, float, float]) -> str:
    """Classify ball color from a small center sample."""
    b, g, r = [float(x) for x in mean_bgr]
    hsv = cv2.cvtColor(np.uint8([[[int(b), int(g), int(r)]]]), cv2.COLOR_BGR2HSV)[0, 0]
    hh, ss, vv = float(hsv[0]), float(hsv[1]), float(hsv[2])

    # Wood / gray rail / neutral noise
    if abs(r - g) < 14 and abs(g - b) < 14 and ss < 55 and 55 < vv < 170:
        return "skip"
    # Bright saturated green cloth (not a ball)
    if 35 <= hh <= 95 and ss > 60 and vv > 110 and g > r + 20 and g > b + 10:
        return "skip"

    if ss < 42 and vv > 185 and abs(r - g) < 25 and abs(g - b) < 25 and min(r, g, b) > 150:
        return "white"
    if vv < 42:
        return "black"
    # Strong red channel (works even under green wash when ball is visible)
    if r > g + 22 and r > b + 15 and r > 90:
        return "red"
    if r > 170 and r > g + 15:
        return "red"
    if r > 140 and r > g + 40:
        return "red"
    # Blue must be B-dominant (avoid gray rail → blue)
    if b > r + 18 and b >= g - 5 and 85 < hh <= 130 and ss > 40:
        return "blue"
    if 148 < hh < 175 and ss > 40 and vv > 95:
        return "pink"
    if 8 < hh <= 28 and ss > 50 and r > b + 5:
        return "brown" if vv < 120 else "yellow"
    if 22 < hh <= 40 and ss > 50 and vv > 105 and r > b:
        return "yellow"
    if 42 < hh <= 80 and ss > 55 and 40 < vv < 95 and g > r + 22:
        return "green"
    # Washed but still clearly reddish
    if r > 130 and r >= g + 8 and vv < 200:
        return "red"
    # Mid-table glare: bright + unsaturated + greenish → skip (not cue ball)
    if vv > 150 and ss < 70 and g >= r - 5:
        return "skip"
    return "skip"


def _sample_ball_color(image_bgr: np.ndarray, cx: float, cy: float, rad: float) -> str:
    """Sample a small center disk to reduce cloth fringe (Dimnir erode idea)."""
    h, w = image_bgr.shape[:2]
    ix, iy = int(cx), int(cy)
    pr = max(1, int(rad * 0.45))
    m = np.zeros((h, w), np.uint8)
    cv2.circle(m, (ix, iy), pr, 255, -1)
    if int(np.count_nonzero(m)) < 3:
        return "skip"
    mean = cv2.mean(image_bgr, mask=m)[:3]
    return _classify_mean_bgr(mean)


def _cluster_score(centers: list[tuple[float, float]], diag: float) -> float:
    if len(centers) < 4:
        return 0.0
    pts = np.array(centers, dtype=np.float32)
    med = np.median(pts, axis=0)
    d = np.linalg.norm(pts - med, axis=1)
    keep_n = max(4, int(len(pts) * 0.7))
    core = pts[np.argsort(d)[:keep_n]]
    mean = core.mean(axis=0)
    spread = float(np.linalg.norm(core - mean, axis=1).mean())
    return max(0.0, 1.0 - (spread / (0.30 * max(diag, 1.0))))


def _order_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as TL, TR, BR, BL."""
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]
    tl = pts[int(np.argmin(s))]
    br = pts[int(np.argmax(s))]
    tr = pts[int(np.argmax(d))]
    bl = pts[int(np.argmin(d))]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _closest_green_quad(image_bgr: np.ndarray) -> dict[str, Any] | None:
    """
    Detect the closest green table as a quadrilateral (trapezoid under CCTV).
    Prefer lower / larger green cloth blobs. Quad is inset so detection
    stays inside the cushion border.
    """
    h, w = image_bgr.shape[:2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    cloth = cv2.inRange(hsv, np.array([35, 50, 40]), np.array([95, 255, 255]))
    cloth = cv2.morphologyEx(cloth, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    cloth = cv2.morphologyEx(cloth, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(cloth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = int(0.06 * h * w)
    best: dict[str, Any] | None = None

    for cnt in cnts:
        area = float(cv2.contourArea(cnt))
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 70 or bh < 70:
            continue
        aspect = max(bw, bh) / max(1, min(bw, bh))
        if aspect > 5.0:
            continue
        cy = y + bh * 0.5
        score = (area / float(h * w)) + 1.6 * (cy / max(h, 1))
        hull = cv2.convexHull(cnt)
        peri = cv2.arcLength(hull, True)
        approx4 = None
        for eps in np.linspace(0.012, 0.12, 28):
            ap = cv2.approxPolyDP(hull, float(eps) * peri, True)
            if len(ap) == 4:
                approx4 = ap.reshape(-1, 2).astype(np.float32)
                break
        box = cv2.boxPoints(cv2.minAreaRect(cnt)).astype(np.float32)
        quad = _order_quad(approx4 if approx4 is not None else box)
        # Strong inset — stay inside cushions / rails (not on players / floor)
        center = quad.mean(axis=0)
        quad = center + (quad - center) * 0.82
        quad[:, 0] = np.clip(quad[:, 0], 0, w - 1)
        quad[:, 1] = np.clip(quad[:, 1], 0, h - 1)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(mask, quad.astype(np.int32), 255)
        # Only green cloth inside the border — kills outside false positives
        mask = cv2.bitwise_and(mask, cloth)
        mask = cv2.erode(mask, np.ones((7, 7), np.uint8), iterations=1)
        if int(np.count_nonzero(mask)) < min_area // 5:
            continue
        cand = {
            "quad": quad,
            "bbox": (int(x), int(y), int(bw), int(bh)),
            "area": int(area),
            "score": float(score),
            "mask": mask,
            "cloth": cloth,
            "contour": cnt,
            "used_approx": approx4 is not None,
        }
        if best is None or cand["score"] > best["score"]:
            best = cand
    return best


def _warp_table(
    image_bgr: np.ndarray, quad: np.ndarray, out_w: int = 420, out_h: int = 840
) -> tuple[np.ndarray, np.ndarray]:
    """Perspective-correct table quad → upright rectangle (outside = black)."""
    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    warped = cv2.warpPerspective(
        image_bgr,
        matrix,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return warped, matrix


def _normalize_table_light(warped_bgr: np.ndarray) -> np.ndarray:
    """Flatten glare / uneven club lighting on the cropped table."""
    # Do not normalize pure-black outside pixels
    valid = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY) > 8
    lab = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(l)
    blur = cv2.GaussianBlur(l, (0, 0), sigmaX=36)
    med = float(np.median(l[valid])) if np.any(valid) else float(np.median(l))
    flat = l.astype(np.float32) - blur.astype(np.float32) + med
    l2 = np.clip(flat, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)
    out = cv2.bilateralFilter(out, d=5, sigmaColor=30, sigmaSpace=30)
    out[~valid] = 0
    return out


def _play_surface_mask(warped_bgr: np.ndarray) -> np.ndarray:
    """
    Hard ROI: only green cloth inside the warped table, with a safety inset.
    Everything else is excluded from detection.
    """
    h, w = warped_bgr.shape[:2]
    # Outer black from warp is already 0; also force a border inset
    inset = np.zeros((h, w), np.uint8)
    y0, y1 = int(h * 0.06), int(h * 0.94)
    x0, x1 = int(w * 0.07), int(w * 0.93)
    inset[y0:y1, x0:x1] = 255

    cloth = _cloth_mask(warped_bgr, for_roi=True)
    # Fill ball holes so the playable bed is solid, then erode to stay inside rails
    bed = cv2.bitwise_and(cloth, inset)
    bed = cv2.morphologyEx(bed, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    bed = cv2.erode(bed, np.ones((9, 9), np.uint8), iterations=2)

    # Keep largest connected green bed only (one table)
    cnts, _ = cv2.findContours(bed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return inset
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.12 * h * w:
        return cv2.bitwise_and(inset, cloth)
    out = np.zeros((h, w), np.uint8)
    cv2.drawContours(out, [c], -1, 255, -1)
    out = cv2.erode(out, np.ones((5, 5), np.uint8), iterations=1)
    return out


def _isolate_table_pixels(warped_bgr: np.ndarray, play_mask: np.ndarray) -> np.ndarray:
    """Black-out everything outside the play surface — detector sees only table."""
    isolated = warped_bgr.copy()
    isolated[play_mask == 0] = 0
    return isolated


def _warped_table_dict(warped: np.ndarray, play_mask: np.ndarray | None = None) -> dict[str, Any]:
    """Table dict whose mask is strictly the playable green surface."""
    h, w = warped.shape[:2]
    mask = play_mask if play_mask is not None else _play_surface_mask(warped)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        x0, y0, bw, bh = int(w * 0.07), int(h * 0.06), int(w * 0.86), int(h * 0.88)
    else:
        x0, y0 = int(xs.min()), int(ys.min())
        bw, bh = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
    return {
        "mask": mask,
        "contour": None,
        "bbox": (x0, y0, bw, bh),
        "centroid": (
            float(xs.mean()) if len(xs) else w * 0.5,
            float(ys.mean()) if len(ys) else h * 0.5,
        ),
        "area": int(np.count_nonzero(mask)),
        "closeness": 1.0,
        "table_key": "near",
        "table_label": "near",
        "closest": True,
        "index": 0,
    }


def find_tables(image_bgr: np.ndarray) -> list[dict[str, Any]]:
    """Separate table regions from cloth contours (closest first)."""
    h, w = image_bgr.shape[:2]
    cloth = _cloth_mask(image_bgr, for_roi=True)
    # Separate nearby tables
    sep = cv2.morphologyEx(cloth, cv2.MORPH_OPEN, np.ones((17, 17), np.uint8))
    contours, _ = cv2.findContours(sep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = int(0.04 * h * w)
    tables: list[dict[str, Any]] = []

    for cnt in contours:
        area = int(cv2.contourArea(cnt))
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 50 or bh < 50:
            continue
        aspect = max(bw, bh) / max(1, min(bw, bh))
        if aspect > 4.0:
            continue
        mask = np.zeros((h, w), np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        # Stay on / near green cloth — drops scoreboard, floor, standing players
        near_cloth = cv2.dilate(_cloth_mask(image_bgr, for_roi=False), np.ones((15, 15), np.uint8), 1)
        mask = cv2.bitwise_and(mask, near_cloth)
        mask = cv2.bitwise_and(mask, cloth)
        # shrink away from cushions / players
        mask = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=1)
        if int(np.count_nonzero(mask)) < min_area // 3:
            continue
        ys, xs = np.where(mask > 0)
        cx, cy = float(xs.mean()), float(ys.mean())
        closeness = (cy / max(h, 1)) * 2.0 + (area / float(h * w))
        tables.append(
            {
                "mask": mask,
                "contour": cnt,
                "bbox": (int(x), int(y), int(bw), int(bh)),
                "centroid": (cx, cy),
                "area": area,
                "closeness": closeness,
            }
        )

    # Optional split: two peaks in horizontal projection
    if len(tables) == 1:
        t = tables[0]
        x, y, bw, bh = t["bbox"]
        if bw > 0.55 * w and bh > 0.25 * h:
            col = (t["mask"] > 0).sum(axis=0).astype(np.float32)
            k = max(5, bw // 20)
            sm = np.convolve(col, np.ones(k, np.float32) / k, mode="same")
            left_b, right_b = x + bw // 3, x + (2 * bw) // 3
            segment = sm[left_b:right_b]
            if segment.size > 0:
                valley = int(left_b + int(np.argmin(segment)))
                lp = float(sm[x:valley].max()) if valley > x else 0.0
                rp = float(sm[valley : x + bw].max()) if valley < x + bw else 0.0
                vv = float(sm[valley])
                if lp > 25 and rp > 25 and vv < 0.4 * min(lp, rp):
                    left, right = t["mask"].copy(), t["mask"].copy()
                    left[:, valley:] = 0
                    right[:, :valley] = 0
                    split = []
                    for m in (left, right):
                        a = int(np.count_nonzero(m))
                        if a < min_area // 3:
                            continue
                        ys, xs = np.where(m > 0)
                        cx, cy = float(xs.mean()), float(ys.mean())
                        split.append(
                            {
                                "mask": m,
                                "contour": t["contour"],
                                "bbox": (
                                    int(xs.min()),
                                    int(ys.min()),
                                    int(xs.max() - xs.min() + 1),
                                    int(ys.max() - ys.min() + 1),
                                ),
                                "centroid": (cx, cy),
                                "area": a,
                                "closeness": (cy / max(h, 1)) * 2.0
                                + (a / float(h * w)),
                            }
                        )
                    if len(split) >= 2:
                        tables = split

    tables.sort(key=lambda t: t["closeness"], reverse=True)
    for idx, t in enumerate(tables):
        cx, cy = t["centroid"]
        gx = min(2, int(cx / max(w, 1) * 3))
        gy = min(2, int(cy / max(h, 1) * 3))
        t["table_key"] = f"t{gx}{gy}"
        t["table_label"] = "near" if idx == 0 else f"far-{idx}"
        t["closest"] = idx == 0
        t["index"] = idx
    return tables


def _objects_on_table(
    image_bgr: np.ndarray,
    cloth: np.ndarray,
    table_mask: np.ndarray,
    *,
    relaxed: bool = False,
) -> np.ndarray:
    """
    Invert soft cloth + color boosts (red / black).
    Avoids Lab 'mild' overlay — that flooded glare into huge false blobs.
    """
    objects = cv2.bitwise_and(cv2.bitwise_not(cloth), table_mask)

    b, g, r = cv2.split(image_bgr.astype(np.float32))
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    # Reds that partially leak into cloth HSV still have R >> G
    rg_min = 8 if relaxed else 12
    r_min = 60 if relaxed else 85
    redish = (
        (r > g + rg_min) & (r > b + 5) & (r > r_min) & (table_mask > 0)
    ).astype(np.uint8) * 255
    black_v = 55 if relaxed else 42
    black = ((hsv[:, :, 2] < black_v) & (table_mask > 0)).astype(np.uint8) * 255

    objects = cv2.bitwise_or(objects, redish)
    objects = cv2.bitwise_or(objects, black)
    objects = cv2.bitwise_and(objects, table_mask)

    objects = cv2.morphologyEx(
        objects, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    )
    objects = cv2.morphologyEx(
        objects, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    return objects


def _filter_contour_dimnir(cnt: np.ndarray, min_a: float, max_a: float, alpha: float = 2.2) -> bool:
    """Dimnir filter_ctrs: size + aspect of min-area rectangle."""
    area = cv2.contourArea(cnt)
    if area < min_a or area > max_a:
        return False
    (_center, (rw, rh), _angle) = cv2.minAreaRect(cnt)
    ww, hh = max(rw, rh), min(rw, rh)
    if hh < 1:
        return False
    if (hh * alpha < ww) or (ww * alpha < hh):
        return False
    return True


def _detect_balls_on_table(
    image_bgr: np.ndarray,
    cloth: np.ndarray,
    table: dict[str, Any],
    *,
    relaxed: bool = False,
) -> list[dict[str, Any]]:
    h, w = image_bgr.shape[:2]
    mask = table["mask"]
    x, y, bw, bh = table["bbox"]
    # Soft cloth keeps ball-sized holes; ROI cloth is only for find_tables
    cloth = _cloth_mask(image_bgr, for_roi=False, wide=relaxed)

    mask_ratio = float(np.count_nonzero(mask)) / float(max(h * w, 1))
    # Warped isolated table already has a tight play-surface mask — light erode only
    if mask_ratio < 0.78:
        core = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=1)
        rim = cv2.subtract(mask, core)
    else:
        core = cv2.erode(mask, np.ones((11, 11), np.uint8), iterations=2)
        yy0, yy1 = y + int(bh * 0.08), y + int(bh * 0.84)
        xx0, xx1 = x + int(bw * 0.05), x + int(bw * 0.95)
        band = np.zeros_like(core)
        band[max(0, yy0) : min(h, yy1), max(0, xx0) : min(w, xx1)] = 255
        core = cv2.bitwise_and(core, band)
        rim = cv2.subtract(mask, cv2.erode(mask, np.ones((15, 15), np.uint8), iterations=1))

    objects = _objects_on_table(image_bgr, cloth, core, relaxed=relaxed)
    objects[rim > 0] = 0
    # Hard rule: never detect outside table mask
    objects[mask == 0] = 0
    if int(np.count_nonzero(objects)) < 8:
        return []

    min_r = max(2, int(min(bw, bh) / 55))
    max_r = max(min_r + 1, min(14, int(min(bw, bh) / 16)))
    min_a = max(6, int(np.pi * min_r * min_r * 0.25))
    max_a = int(np.pi * max_r * max_r * 2.5)
    min_peak_dist = max(4.5 if relaxed else 5.0, min_r * (1.35 if relaxed else 1.5))
    ring_min = 0.48 if relaxed else 0.62
    red_rg = 12 if relaxed else 18
    red_abs = 75 if relaxed else 110
    rg_thr = 16 if relaxed else 25
    rg_r_min = 70 if relaxed else 100

    balls: list[dict[str, Any]] = []
    used: list[tuple[float, float]] = []

    def cloth_ring_ok(cx: float, cy: float, rad: float) -> bool:
        """Ball sits on cloth: annulus around center should be mostly green."""
        ix, iy = int(cx), int(cy)
        r0 = max(3, int(rad * 1.3))
        r1 = max(r0 + 2, int(max(rad * 2.8, 8)))
        ring = np.zeros((h, w), np.uint8)
        cv2.circle(ring, (ix, iy), r1, 255, -1)
        cv2.circle(ring, (ix, iy), r0, 0, -1)
        ring = cv2.bitwise_and(ring, mask)
        n = int(np.count_nonzero(ring))
        if n < 12:
            return False
        cloth_hit = int(np.count_nonzero(cv2.bitwise_and(ring, cloth)))
        return (cloth_hit / n) >= ring_min

    def try_add(cx: float, cy: float, rad: float, source: str) -> None:
        ix, iy = int(cx), int(cy)
        if ix < 4 or iy < 4 or ix >= w - 4 or iy >= h - 4:
            return
        if core[iy, ix] == 0 or mask[iy, ix] == 0:
            return
        if rim[iy, ix] > 0:
            return
        if rad < 1.1 or rad > max_r * 1.8:
            return
        if any((cx - ux) ** 2 + (cy - uy) ** 2 < min_peak_dist**2 for ux, uy in used):
            return
        if not cloth_ring_ok(cx, cy, rad):
            return
        color = _sample_ball_color(image_bgr, cx, cy, max(rad, 3.0))
        if color == "skip":
            return
        # Glare / shirt highlights often classify as white with large radius
        if color == "white" and rad > 5.5:
            return
        if color in ("blue", "yellow", "green", "brown", "pink") and rad > 7.5:
            return
        # Oversized "red" blobs are usually glare / clothing, not balls
        if color == "red" and rad > 7.0:
            return
        # Confirm reds with R-G at the sample (kills washed glare FPs)
        if color == "red":
            pr = max(2, int(max(rad, 3.0) * 0.45))
            patch = image_bgr[
                max(0, iy - pr) : iy + pr + 1, max(0, ix - pr) : ix + pr + 1
            ]
            if patch.size == 0:
                return
            mb, mg, mr = [float(v) for v in patch.reshape(-1, 3).mean(0)]
            if mr < mg + red_rg or mr < red_abs:
                return
        balls.append({"xy": (cx, cy), "r": float(rad), "color": color, "source": source})
        used.append((cx, cy))

    def peaks_from_mask(bin_mask: np.ndarray, source: str, thr_frac: float = 0.22) -> None:
        if int(np.count_nonzero(bin_mask)) < 5:
            return
        dist = cv2.distanceTransform(bin_mask, cv2.DIST_L2, 5)
        dmax = float(dist.max())
        if dmax < 1.0:
            return
        # Cap dmax so one glare blob doesn't raise thr too high
        thr = max(1.0 if relaxed else 1.15, thr_frac * min(dmax, 10.0))
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        local_max = (dist == cv2.dilate(dist, ker)) & (dist >= thr)
        pts = np.column_stack(np.where(local_max))
        pts = sorted(pts, key=lambda p: float(dist[p[0], p[1]]), reverse=True)
        for py, px in pts:
            try_add(float(px), float(py), float(dist[py, px]), source)

    peaks_from_mask(objects, "dist", thr_frac=0.16 if relaxed else 0.20)

    cnts, _ = cv2.findContours(objects, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    typical = max(min_a * 1.6, np.pi * (min_r * 1.1) ** 2)
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < min_a:
            continue
        (_c, (rw, rh), _a) = cv2.minAreaRect(cnt)
        ww, hh = max(rw, rh), min(rw, rh)
        if hh < 1 or ww > 2.8 * hh:
            continue
        M = cv2.moments(cnt)
        if M["m00"] < 1:
            continue
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        (_center, r) = cv2.minEnclosingCircle(cnt)

        if area <= max_a and _filter_contour_dimnir(cnt, min_a, max_a, alpha=2.4):
            peri = cv2.arcLength(cnt, True)
            circ = 4 * np.pi * area / (peri * peri + 1e-6)
            if circ >= (0.18 if relaxed else 0.22):
                try_add(cx, cy, float(r), "contour")
                continue

        if area > typical * 1.5:
            local = np.zeros((h, w), np.uint8)
            cv2.drawContours(local, [cnt], -1, 255, -1)
            local = cv2.bitwise_and(local, objects)
            peaks_from_mask(local, "split", thr_frac=0.18 if relaxed else 0.22)

    # Dedicated strong-red peaks (R-G) — most reliable under CCTV green wash
    bch, gch, rch = cv2.split(image_bgr.astype(np.float32))
    rg = (rch - gch)
    rg[core == 0] = -999
    rg_bin = ((rg > rg_thr) & (rch > rg_r_min) & (core > 0)).astype(np.uint8) * 255
    rg_bin = cv2.morphologyEx(rg_bin, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    peaks_from_mask(rg_bin, "redpeak", thr_frac=0.15 if relaxed else 0.18)

    # Keep reds first (up to MAX_REDS), then best other colors — avoids glare
    # whites crowding out real reds when len > MAX_BALLS.
    reds = [b for b in balls if b["color"] == "red"]
    others = [b for b in balls if b["color"] != "red"]
    if len(reds) > MAX_REDS:
        pts = np.array([b["xy"] for b in reds], dtype=np.float32)
        scored = []
        for i, b in enumerate(reds):
            d = np.linalg.norm(pts - pts[i], axis=1)
            neigh = int(np.sum(d < 0.14 * ((bw**2 + bh**2) ** 0.5)))
            scored.append((neigh, -b["r"], i))
        scored.sort(reverse=True)
        keep = {i for _, __, i in scored[:MAX_REDS]}
        reds = [b for i, b in enumerate(reds) if i in keep]
    room = max(0, MAX_BALLS - len(reds))
    if len(others) > room:
        pts = np.array([b["xy"] for b in others], dtype=np.float32) if others else np.zeros((0, 2))
        scored = []
        for i, b in enumerate(others):
            d = np.linalg.norm(pts - pts[i], axis=1) if len(pts) else np.array([0.0])
            neigh = int(np.sum(d < 0.16 * ((bw**2 + bh**2) ** 0.5)))
            bonus = {
                "white": 3,
                "blue": 3,
                "black": 3,
                "yellow": 2,
                "pink": 2,
                "brown": 2,
                "green": 1,
            }.get(b["color"], 0)
            cy = b["xy"][1]
            ypen = 0 if (y + bh * 0.12) < cy < (y + bh * 0.82) else -4
            scored.append((neigh + bonus + ypen, -b["r"], i))
        scored.sort(reverse=True)
        keep = {i for _, __, i in scored[:room]}
        others = [b for i, b in enumerate(others) if i in keep]
    return reds + others


def _count_colors(balls: list[dict[str, Any]]) -> BallCounts:
    counts = BallCounts()
    for b in balls:
        c = b["color"]
        if c == "red" and counts.red < MAX_REDS:
            counts.red += 1
        elif c == "white" and counts.white < 1:
            counts.white += 1
        elif c == "yellow" and counts.yellow < 1:
            counts.yellow += 1
        elif c == "green" and counts.green < 1:
            counts.green += 1
        elif c == "brown" and counts.brown < 1:
            counts.brown += 1
        elif c == "blue" and counts.blue < 1:
            counts.blue += 1
        elif c == "pink" and counts.pink < 1:
            counts.pink += 1
        elif c == "black" and counts.black < 1:
            counts.black += 1
    return counts


def _suggest_state(
    cloth_ratio: float, counts: BallCounts, detections: int, cluster: float
) -> tuple[str, bool, bool]:
    reds = counts.red
    racked = (
        (reds >= 8 and cluster >= 0.2)
        or (reds >= 10)
        or (detections >= 11 and reds >= 7 and cluster >= 0.15)
    )
    cleared = reds <= 1 and detections <= 3
    mid = detections >= 3 and not racked

    if cloth_ratio < 0.02 and detections == 0:
        return "NO_TABLE", False, False
    if racked:
        return "RACKED", True, False
    if cleared:
        return "CLEARED", False, True
    if mid:
        return "IN_PLAY", False, False
    return "UNKNOWN", False, False


def analyze_table(
    image_bgr: np.ndarray,
    cloth: np.ndarray,
    table: dict[str, Any],
    prev_gray: np.ndarray | None,
    full_cloth_ratio: float,
    *,
    relaxed: bool = False,
) -> dict[str, Any]:
    h, w = image_bgr.shape[:2]
    x, y, bw, bh = table["bbox"]
    diag = float((bw**2 + bh**2) ** 0.5)
    balls = _detect_balls_on_table(image_bgr, cloth, table, relaxed=relaxed)
    counts = _count_colors(balls)
    red_centers = [b["xy"] for b in balls if b["color"] == "red"]
    centers = [b["xy"] for b in balls]
    cluster = _cluster_score(red_centers or centers, diag)
    # Snooker total = reds + uniquely counted colors (not raw blob count)
    detections = counts.total()
    sources = sorted({b["source"] for b in balls})
    engine = "sbt+" + "+".join(sources) if sources else "sbt"
    if relaxed:
        engine = engine + "+relaxed"

    suggested, racked, cleared = _suggest_state(
        full_cloth_ratio, counts, detections, cluster
    )

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    motion = 0.0
    if prev_gray is not None:
        if prev_gray.shape != gray.shape:
            prev_gray = cv2.resize(prev_gray, (w, h))
        diff = cv2.absdiff(prev_gray, gray)
        m = table["mask"] > 0
        if np.any(m):
            motion = float(np.mean(diff[m]) / 255.0)

    ball_rows = [
        {
            "x": round(float(b["xy"][0]), 1),
            "y": round(float(b["xy"][1]), 1),
            "r": round(float(b["r"]), 1),
            "color": b["color"],
        }
        for b in balls
    ]

    return {
        "table_key": table["table_key"],
        "table_label": table["table_label"],
        "closest": bool(table["closest"]),
        "table_index": table["index"],
        "bbox": table["bbox"],
        "cloth_ratio": round(full_cloth_ratio, 4),
        "counts": asdict(counts),
        "red_cluster_score": round(float(cluster), 4),
        "motion_score": round(motion, 4),
        "suggested_state": suggested,
        "racked": racked,
        "cleared": cleared,
        "frame_size": {"w": w, "h": h},
        "engine": engine,
        "detections": detections,
        "balls": ball_rows,
        "model_error": None,
    }



def _prep_working_frame(image_bgr: np.ndarray) -> dict[str, Any]:
    """Resize + quality + optional CCTV preprocess. No ball detection."""
    h0, w0 = image_bgr.shape[:2]
    quality = _frame_quality(image_bgr)
    target = 720
    long_side = max(w0, h0)
    img = image_bgr
    if long_side != target:
        scale = target / max(long_side, 1)
        img = cv2.resize(image_bgr, (int(w0 * scale), int(h0 * scale)))
    h, w = img.shape[:2]
    no_signal = float(img.mean()) < 8
    if not no_signal:
        img = _preprocess_cctv(img, quality)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {
        "image": img,
        "gray": gray,
        "quality": quality,
        "no_signal": no_signal,
        "frame_size": {"w": w, "h": h},
    }


def _cheap_motion(
    gray: np.ndarray,
    prev_gray: np.ndarray | None,
    roi_mask: np.ndarray | None = None,
) -> float:
    """
    Fast downsampled absdiff on table ROI (or full frame).
    Uses top ~8% of pixels so local player/cue motion isn't washed out by mean.
    """
    if prev_gray is None:
        return 0.0
    g, pg = gray, prev_gray
    if pg.shape != g.shape:
        pg = cv2.resize(pg, (g.shape[1], g.shape[0]))
    g_s = cv2.resize(g, (0, 0), fx=0.35, fy=0.35, interpolation=cv2.INTER_AREA)
    pg_s = cv2.resize(pg, (g_s.shape[1], g_s.shape[0]), interpolation=cv2.INTER_AREA)
    diff = cv2.absdiff(pg_s, g_s)
    if roi_mask is not None and roi_mask.size:
        m = roi_mask
        if m.shape != g.shape:
            m = cv2.resize(m, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
        m_s = cv2.resize(
            m, (g_s.shape[1], g_s.shape[0]), interpolation=cv2.INTER_NEAREST
        )
        sel = m_s > 0
        if np.any(sel):
            vals = diff[sel].astype(np.float32)
        else:
            vals = diff.reshape(-1).astype(np.float32)
    else:
        vals = diff.reshape(-1).astype(np.float32)
    if vals.size == 0:
        return 0.0
    k = max(8, int(vals.size * 0.08))
    if k >= vals.size:
        top = vals
    else:
        top = np.partition(vals, -k)[-k:]
    return float(np.mean(top) / 255.0)


def analyze_frame(image_bgr: np.ndarray, prev_gray: np.ndarray | None = None) -> dict[str, Any]:
    prep = _prep_working_frame(image_bgr)
    image_bgr = prep["image"]
    quality = prep["quality"]
    gray = prep["gray"]
    h, w = gray.shape[:2]

    if prep["no_signal"]:
        return {
            "cloth_ratio": 0.0,
            "counts": asdict(BallCounts()),
            "red_cluster_score": 0.0,
            "motion_score": 0.0,
            "suggested_state": "NO_SIGNAL",
            "racked": False,
            "cleared": False,
            "frame_size": {"w": w, "h": h},
            "engine": "none",
            "detections": 0,
            "balls": [],
            "tables": [],
            "table_label": None,
            "note": "Black frame — camera/channel has no video",
            "quality": quality,
            "gray": gray,
        }

    green = _closest_green_quad(image_bgr)
    if green is not None:
        warped, _matrix = _warp_table(image_bgr, green["quad"])
        warped = _normalize_table_light(warped)
        play_mask = _play_surface_mask(warped)
        warped = _isolate_table_pixels(warped, play_mask)
        wh, ww = warped.shape[:2]
        cloth_w = _cloth_mask(
            warped, wide=bool(quality.get("poor") or quality.get("low_res"))
        )
        play_n = max(int(np.count_nonzero(play_mask)), 1)
        cloth_ratio = float(np.count_nonzero(cv2.bitwise_and(cloth_w, play_mask))) / float(
            play_n
        )
        table_w = _warped_table_dict(warped, play_mask)
        use_relaxed = bool(
            quality.get("dark") or quality.get("blurry") or quality.get("low_res")
        )
        primary = analyze_table(
            warped, cloth_w, table_w, None, cloth_ratio, relaxed=use_relaxed
        )
        if (
            not use_relaxed
            and cloth_ratio > 0.15
            and int((primary.get("counts") or {}).get("red") or 0) < 6
        ):
            retry = analyze_table(
                warped, cloth_w, table_w, None, cloth_ratio, relaxed=True
            )
            if int((retry.get("counts") or {}).get("red") or 0) > int(
                (primary.get("counts") or {}).get("red") or 0
            ):
                primary = retry

        motion = 0.0
        if prev_gray is not None:
            pg = prev_gray
            if pg.shape != gray.shape:
                pg = cv2.resize(pg, (w, h))
            diff = cv2.absdiff(pg, gray)
            m = green["mask"] > 0
            if np.any(m):
                motion = float(np.mean(diff[m]) / 255.0)
        primary["motion_score"] = round(motion, 4)
        primary["bbox"] = green["bbox"]
        primary["quad"] = green["quad"].reshape(-1).tolist()
        primary["pipeline"] = "crop+warp+norm+inside"
        primary["engine"] = (primary.get("engine") or "sbt") + "+warp+inside"
        primary["frame_size"] = {"w": w, "h": h}
        primary["warped_size"] = {"w": ww, "h": wh}
        primary["roi_pixels"] = int(np.count_nonzero(play_mask))
        primary["table_mask"] = green["mask"]
        row = {k: v for k, v in primary.items() if k != "table_mask"}
        primary["tables"] = [row]
        primary["table_count"] = 1
        primary["quality"] = quality
        primary["gray"] = gray
        primary["cloth_ratio"] = round(cloth_ratio, 4)
        return primary

    cloth = _cloth_mask(image_bgr, wide=bool(quality.get("poor") or quality.get("low_res")))
    cloth_ratio = float(np.count_nonzero(cloth)) / float(h * w)
    tables = find_tables(image_bgr)

    if not tables:
        return {
            "cloth_ratio": round(cloth_ratio, 4),
            "counts": asdict(BallCounts()),
            "red_cluster_score": 0.0,
            "motion_score": 0.0,
            "suggested_state": "NO_TABLE",
            "racked": False,
            "cleared": False,
            "frame_size": {"w": w, "h": h},
            "engine": "none",
            "detections": 0,
            "balls": [],
            "tables": [],
            "table_label": None,
            "quality": quality,
            "gray": gray,
        }

    use_relaxed = bool(quality.get("dark") or quality.get("blurry"))
    closest = tables[0]
    primary = analyze_table(
        image_bgr, cloth, closest, prev_gray, cloth_ratio, relaxed=use_relaxed
    )
    if (
        not use_relaxed
        and cloth_ratio > 0.12
        and int((primary.get("counts") or {}).get("red") or 0) < 6
    ):
        retry = analyze_table(
            image_bgr, cloth, closest, prev_gray, cloth_ratio, relaxed=True
        )
        if int((retry.get("counts") or {}).get("red") or 0) > int(
            (primary.get("counts") or {}).get("red") or 0
        ):
            primary = retry

    primary = dict(primary)
    primary["pipeline"] = "mask-fallback"
    primary["table_mask"] = closest.get("mask")
    row = {k: v for k, v in primary.items() if k != "table_mask"}
    primary["tables"] = [row]
    primary["table_count"] = 1
    primary["quality"] = quality
    primary["gray"] = gray
    return primary


class _BallTracker:
    """Greedy centroid tracker on warped-table detections."""

    MATCH_PX = 48.0
    STOP_SPEED = 7.0

    def __init__(self) -> None:
        self.tracks: list[dict[str, Any]] = []
        self._next_id = 1

    def reset(self) -> None:
        self.tracks = []
        self._next_id = 1

    def update(self, balls: list[dict[str, Any]]) -> dict[str, Any]:
        dets = [
            {
                "xy": (float(b["x"]), float(b["y"])),
                "color": str(b.get("color") or "unknown"),
            }
            for b in (balls or [])
            if "x" in b and "y" in b
        ]
        used: set[int] = set()
        matched: list[dict[str, Any]] = []

        for tr in self.tracks:
            best_i, best_d = -1, self.MATCH_PX
            tx, ty = tr["xy"]
            for i, d in enumerate(dets):
                if i in used:
                    continue
                dx = d["xy"][0] - tx
                dy = d["xy"][1] - ty
                dist = float((dx * dx + dy * dy) ** 0.5)
                if dist < best_d:
                    best_d, best_i = dist, i
            if best_i < 0:
                tr = dict(tr)
                tr["missed"] = int(tr.get("missed") or 0) + 1
                tr["speed"] = 0.0
                if tr["missed"] <= 3:
                    matched.append(tr)
                continue
            used.add(best_i)
            d = dets[best_i]
            matched.append(
                {
                    "id": tr["id"],
                    "xy": d["xy"],
                    "color": d["color"],
                    "missed": 0,
                    "speed": best_d,
                }
            )

        for i, d in enumerate(dets):
            if i in used:
                continue
            matched.append(
                {
                    "id": self._next_id,
                    "xy": d["xy"],
                    "color": d["color"],
                    "missed": 0,
                    "speed": 0.0,
                }
            )
            self._next_id += 1

        self.tracks = matched
        speeds = [float(t["speed"]) for t in self.tracks if int(t.get("missed") or 0) == 0]
        mean_speed = float(np.mean(speeds)) if speeds else 0.0
        moving = sum(1 for s in speeds if s >= self.STOP_SPEED)
        return {
            "track_count": len(self.tracks),
            "mean_speed": round(mean_speed, 2),
            "moving_balls": int(moving),
            "balls_stopped": moving == 0 and mean_speed < self.STOP_SPEED,
        }


class _TableSM:
    """Commit counts on settle; emit rack/start/end only when allowed."""

    HIST = 7

    def __init__(self) -> None:
        self.state = "IDLE"
        self.rack_hits = 0
        self.clear_hits = 0
        self.motion_hits = 0
        self.frame_id = 0
        self._reds: list[int] = []
        self._dets: list[int] = []
        self._states: list[str] = []
        self._clusters: list[float] = []
        self.stable_reds = 0
        self.stable_dets = 0
        self._have_stable = False

    @staticmethod
    def _median_int(vals: list[int]) -> int:
        if not vals:
            return 0
        s = sorted(vals)
        return int(s[len(s) // 2])

    @staticmethod
    def _majority(vals: list[str]) -> str:
        if not vals:
            return "UNKNOWN"
        from collections import Counter

        return Counter(vals).most_common(1)[0][0]

    def commit_counts(self, analysis: dict[str, Any]) -> dict[str, Any]:
        frame_reds = int(analysis.get("counts", {}).get("red") or 0)
        frame_dets = int(analysis.get("detections") or 0)
        raw_state = str(analysis.get("suggested_state") or "UNKNOWN")
        raw_cl = float(analysis.get("red_cluster_score") or 0)

        self._reds.append(frame_reds)
        self._dets.append(frame_dets)
        self._states.append(raw_state)
        self._clusters.append(raw_cl)
        for buf in (self._reds, self._dets, self._states, self._clusters):
            if len(buf) > self.HIST:
                del buf[0 : len(buf) - self.HIST]

        med_reds = self._median_int(self._reds)
        med_dets = self._median_int(self._dets)
        if len(self._reds) >= 3:
            hi_reds = sorted(self._reds)[-2]
            hi_dets = sorted(self._dets)[-2]
            med_reds = max(med_reds, (med_reds + hi_reds) // 2)
            med_dets = max(med_dets, (med_dets + hi_dets) // 2)

        if not self._have_stable:
            self.stable_reds, self.stable_dets = med_reds, med_dets
            self._have_stable = True
        else:
            step = 2
            self.stable_reds += max(-step, min(step, med_reds - self.stable_reds))
            self.stable_dets += max(-step, min(step, med_dets - self.stable_dets))

        maj = self._majority(self._states)
        cl = float(sorted(self._clusters)[len(self._clusters) // 2])
        counts = dict(analysis.get("counts") or {})
        counts["red"] = min(MAX_REDS, max(0, self.stable_reds))
        out = dict(analysis)
        out["counts"] = counts
        out["detections"] = min(MAX_BALLS, max(0, self.stable_dets))
        out["raw_detections"] = frame_dets
        out["raw_reds"] = frame_reds
        out["red_cluster_score"] = round(cl, 4)
        out["suggested_state"] = maj
        out["racked"] = maj == "RACKED"
        out["cleared"] = maj == "CLEARED"
        out["stable"] = len(self._reds) >= 2
        out.pop("table_mask", None)
        out.pop("gray", None)
        return out

    def overlay_stable(self, analysis: dict[str, Any]) -> dict[str, Any]:
        out = dict(analysis)
        out.pop("table_mask", None)
        out.pop("gray", None)
        if not self._have_stable:
            return out
        counts = dict(out.get("counts") or {})
        out["raw_reds"] = int(counts.get("red") or 0)
        out["raw_detections"] = int(out.get("detections") or 0)
        counts["red"] = min(MAX_REDS, max(0, self.stable_reds))
        out["counts"] = counts
        out["detections"] = min(MAX_BALLS, max(0, self.stable_dets))
        maj = self._majority(self._states) if self._states else out.get("suggested_state")
        out["suggested_state"] = maj
        out["racked"] = maj == "RACKED"
        out["cleared"] = maj == "CLEARED"
        out["stable"] = False
        out["count_frozen"] = True
        return out

    def evaluate_events(
        self, analysis: dict[str, Any], *, motion_active: bool
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        events: list[dict[str, Any]] = []
        label = analysis.get("table_label") or "table"
        key = analysis.get("table_key")

        if analysis.get("suggested_state") in ("NO_SIGNAL", "NO_TABLE"):
            return events, analysis

        reds = int((analysis.get("counts") or {}).get("red") or 0)
        dets = int(analysis.get("detections") or 0)
        cl = float(analysis.get("red_cluster_score") or 0)
        if reds >= 9 and cl >= 0.2:
            suggested = "RACKED"
        elif reds >= 10:
            suggested = "RACKED"
        elif reds <= 1 and dets <= 3:
            suggested = "CLEARED"
        elif dets >= 3:
            suggested = "IN_PLAY"
        else:
            suggested = str(analysis.get("suggested_state") or "UNKNOWN")
        analysis = dict(analysis)
        analysis["suggested_state"] = suggested
        analysis["racked"] = suggested == "RACKED"
        analysis["cleared"] = suggested == "CLEARED"

        if motion_active:
            self.motion_hits = min(10, self.motion_hits + 2)
        else:
            self.motion_hits = max(0, self.motion_hits - 1)

        if suggested == "RACKED":
            self.rack_hits += 1
            self.clear_hits = 0
        else:
            self.rack_hits = max(0, self.rack_hits - 1)

        if suggested == "CLEARED":
            self.clear_hits += 1
        else:
            self.clear_hits = max(0, self.clear_hits - 1)

        if (
            self.state in ("IDLE", "FRAME_ENDED")
            and self.rack_hits >= 2
            and analysis.get("stable")
        ):
            self.state = "RACKED"
            self.frame_id += 1
            events.append(
                {
                    "type": "RACK_DETECTED",
                    "message": (
                        f"[{label}] Rack stable "
                        f"(balls≈{analysis.get('detections')}, reds≈{reds})"
                    ),
                    "frame_id": self.frame_id,
                    "table_key": key,
                    "table_label": label,
                }
            )

        if self.state == "RACKED" and (
            (self.motion_hits >= 2 and not analysis["racked"])
            or (suggested == "IN_PLAY" and self.motion_hits >= 1 and reds <= 8)
        ):
            events.append(
                {
                    "type": "FRAME_STARTED",
                    "message": f"[{label}] Break/motion — frame started",
                    "frame_id": self.frame_id,
                    "table_key": key,
                    "table_label": label,
                }
            )
            self.state = "IN_PLAY"
            events.append(
                {
                    "type": "IN_PLAY",
                    "message": f"[{label}] Frame in play",
                    "frame_id": self.frame_id,
                    "table_key": key,
                    "table_label": label,
                }
            )

        if self.state == "IN_PLAY" and self.clear_hits >= 3:
            self.state = "FRAME_ENDED"
            events.append(
                {
                    "type": "FRAME_ENDED",
                    "message": f"[{label}] Table cleared — frame ended",
                    "frame_id": self.frame_id,
                    "table_key": key,
                    "table_label": label,
                }
            )
            self.rack_hits = self.clear_hits = self.motion_hits = 0

        return events, analysis


class FrameStateMachine:
    """
    Best-case live loop:
      WATCHING → cheap motion only
      ACTIVE   → detect + track (UI counts frozen)
      settle   → commit counts + evaluate events → WATCHING
    """

    MOTION_ON = 0.026
    MOTION_OFF = 0.016
    STOP_NEEDED = 2
    FORCE_EVERY = 18

    def __init__(self) -> None:
        self.tables: dict[str, _TableSM] = {}
        self.tracker = _BallTracker()
        self.prev_gray: np.ndarray | None = None
        self.roi_mask: np.ndarray | None = None
        self.last_analysis: dict[str, Any] | None = None
        self.phase = "IDLE"  # IDLE | WATCHING | ACTIVE
        self.stop_hits = 0
        self.samples = 0
        self.samples_since_detect = 0
        self.state = "IDLE"
        self.frame_id = 0
        self.episode_motion = False

    def _annotate(
        self,
        analysis: dict[str, Any],
        *,
        motion: float,
        detector_ran: bool,
        track_info: dict[str, Any] | None,
        settled_now: bool,
        phase: str,
    ) -> dict[str, Any]:
        out = {k: v for k, v in analysis.items() if k not in ("gray", "table_mask")}
        out["motion_score"] = round(float(motion), 4)
        out["detector_ran"] = detector_ran
        out["phase"] = phase
        out["settled"] = bool(settled_now or phase == "WATCHING")
        out["count_frozen"] = phase == "ACTIVE"
        if track_info:
            out.update(track_info)
        out["stop_hits"] = self.stop_hits
        out["samples_since_detect"] = self.samples_since_detect
        return out

    def update(self, image_bgr: np.ndarray) -> dict[str, Any]:
        self.samples += 1
        events: list[dict[str, Any]] = []

        prep = _prep_working_frame(image_bgr)
        gray = prep["gray"]
        motion = _cheap_motion(gray, self.prev_gray, self.roi_mask)
        self.prev_gray = gray
        self.samples_since_detect += 1

        if prep["no_signal"]:
            self.phase = "IDLE"
            analysis = {
                "cloth_ratio": 0.0,
                "counts": asdict(BallCounts()),
                "red_cluster_score": 0.0,
                "suggested_state": "NO_SIGNAL",
                "racked": False,
                "cleared": False,
                "frame_size": prep["frame_size"],
                "engine": "none",
                "detections": 0,
                "balls": [],
                "tables": [],
                "note": "Black frame — camera/channel has no video",
                "quality": prep["quality"],
            }
            analysis = self._annotate(
                analysis,
                motion=motion,
                detector_ran=False,
                track_info=None,
                settled_now=False,
                phase=self.phase,
            )
            self.last_analysis = analysis
            return {"state": self.state, "events": events, "analysis": analysis}

        force = (
            self.last_analysis is None
            or self.phase == "IDLE"
            or self.samples_since_detect >= self.FORCE_EVERY
        )
        moved = motion >= self.MOTION_ON
        quiet = motion < self.MOTION_OFF

        if force:
            run_detect = True
        elif self.phase == "WATCHING":
            if moved:
                self.phase = "ACTIVE"
                self.episode_motion = True
                self.stop_hits = 0
                run_detect = True
            else:
                run_detect = False
        elif self.phase == "ACTIVE":
            run_detect = True
        else:
            run_detect = True

        if not run_detect and self.last_analysis is not None:
            analysis = self._annotate(
                dict(self.last_analysis),
                motion=motion,
                detector_ran=False,
                track_info={
                    "track_count": len(self.tracker.tracks),
                    "mean_speed": 0.0,
                    "moving_balls": 0,
                    "balls_stopped": True,
                },
                settled_now=False,
                phase="WATCHING",
            )
            analysis["count_frozen"] = False
            analysis["skipped_detect"] = True
            self.last_analysis = analysis
            return {"state": self.state, "events": events, "analysis": analysis}

        result = analyze_frame(image_bgr, None)
        result.pop("gray", None)
        mask = result.pop("table_mask", None)
        if mask is not None:
            self.roi_mask = mask
        elif result.get("bbox"):
            x, y, bw, bh = [int(v) for v in result["bbox"]]
            hh, ww = gray.shape[:2]
            m = np.zeros((hh, ww), dtype=np.uint8)
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(ww, x + bw), min(hh, y + bh)
            if x1 > x0 and y1 > y0:
                m[y0:y1, x0:x1] = 255
                self.roi_mask = m

        self.samples_since_detect = 0
        track_info = self.tracker.update(result.get("balls") or [])
        balls_stopped = bool(track_info.get("balls_stopped"))

        if result.get("suggested_state") == "NO_TABLE":
            self.phase = "IDLE"
            self.tracker.reset()
            analysis = self._annotate(
                result,
                motion=motion,
                detector_ran=True,
                track_info=track_info,
                settled_now=False,
                phase=self.phase,
            )
            self.last_analysis = analysis
            return {"state": self.state, "events": events, "analysis": analysis}

        if moved and self.phase != "ACTIVE":
            self.phase = "ACTIVE"
            self.episode_motion = True
            self.stop_hits = 0

        settled_now = False
        if self.phase == "ACTIVE":
            if quiet and balls_stopped:
                self.stop_hits += 1
            else:
                self.stop_hits = 0
            if self.stop_hits >= self.STOP_NEEDED:
                settled_now = True
                self.phase = "WATCHING"
        else:
            if quiet:
                settled_now = True
                self.phase = "WATCHING"
            else:
                self.phase = "ACTIVE"
                self.episode_motion = True
                self.stop_hits = 0

        key = result.get("table_key") or "near"
        sm = self.tables.setdefault(key, _TableSM())

        if not sm._have_stable:
            seeded = sm.commit_counts(dict(result))
            if self.phase == "ACTIVE" and not settled_now:
                analysis = self._annotate(
                    sm.overlay_stable(seeded),
                    motion=motion,
                    detector_ran=True,
                    track_info=track_info,
                    settled_now=False,
                    phase=self.phase,
                )
            else:
                if settled_now:
                    ev, seeded = sm.evaluate_events(
                        seeded, motion_active=self.episode_motion
                    )
                    events.extend(ev)
                    self.episode_motion = False
                analysis = self._annotate(
                    seeded,
                    motion=motion,
                    detector_ran=True,
                    track_info=track_info,
                    settled_now=settled_now,
                    phase=self.phase,
                )
                analysis["count_frozen"] = self.phase == "ACTIVE"
        elif settled_now:
            committed = sm.commit_counts(dict(result))
            ev, committed = sm.evaluate_events(
                committed, motion_active=self.episode_motion
            )
            events.extend(ev)
            self.episode_motion = False
            analysis = self._annotate(
                committed,
                motion=motion,
                detector_ran=True,
                track_info=track_info,
                settled_now=True,
                phase=self.phase,
            )
            analysis["count_frozen"] = False
        else:
            held = sm.overlay_stable(dict(result))
            analysis = self._annotate(
                held,
                motion=motion,
                detector_ran=True,
                track_info=track_info,
                settled_now=False,
                phase=self.phase,
            )

        row = {k: v for k, v in analysis.items() if k != "tables"}
        analysis = dict(analysis)
        analysis["tables"] = [row]
        analysis["table_count"] = 1
        self.last_analysis = analysis
        self.state = sm.state
        self.frame_id = sm.frame_id
        return {"state": self.state, "events": events, "analysis": analysis}
