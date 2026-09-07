/**
 * SCOS detector adapter interface (Phase 2B).
 *
 * Implementations must:
 *   - accept a perspective-corrected JPEG buffer
 *   - return SCOS-normalized detections
 *   - never touch camera / calibration / scoring
 *
 * Normalized detection:
 *   { class, confidence, x, y, width, height, raw_class?, model_class_id? }
 * where x,y = top-left of the box in warped-table pixels.
 */

export const SCOS_BALL_CLASSES = [
  'red',
  'yellow',
  'green',
  'brown',
  'blue',
  'pink',
  'black',
  'cue_ball',
];

/** Map common pretrained label strings → SCOS class names. Unmapped stay as raw. */
const CLASS_ALIASES = {
  red: 'red',
  reds: 'red',
  'red ball': 'red',
  red_ball: 'red',
  yellow: 'yellow',
  'yellow ball': 'yellow',
  yellow_ball: 'yellow',
  green: 'green',
  'green ball': 'green',
  green_ball: 'green',
  brown: 'brown',
  'brown ball': 'brown',
  brown_ball: 'brown',
  blue: 'blue',
  'blue ball': 'blue',
  blue_ball: 'blue',
  pink: 'pink',
  'pink ball': 'pink',
  pink_ball: 'pink',
  black: 'black',
  'black ball': 'black',
  black_ball: 'black',
  white: 'cue_ball',
  'white ball': 'cue_ball',
  white_ball: 'cue_ball',
  cue: 'cue_ball',
  cueball: 'cue_ball',
  cue_ball: 'cue_ball',
  'cue ball': 'cue_ball',
  // Numeric IDs used by snooker-ball-detection-rnhxo family (0=white … 7=black)
  '0': 'cue_ball',
  '1': 'red',
  '2': 'yellow',
  '3': 'green',
  '4': 'brown',
  '5': 'blue',
  '6': 'pink',
  '7': 'black',
};

export function normalizeClassName(raw) {
  if (raw == null) return 'unknown';
  const key = String(raw).trim().toLowerCase();
  return CLASS_ALIASES[key] || key.replace(/\s+/g, '_');
}

/**
 * Convert a Roboflow-style center-box prediction into SCOS top-left box.
 * Roboflow: x,y = center; width,height = size (pixels of the inference image).
 */
export function fromCenterBox(pred, { scaleX = 1, scaleY = 1 } = {}) {
  const cx = Number(pred.x) * scaleX;
  const cy = Number(pred.y) * scaleY;
  const w = Number(pred.width) * scaleX;
  const h = Number(pred.height) * scaleY;
  const raw = pred.class ?? pred.class_name ?? pred.label ?? 'unknown';
  return {
    class: normalizeClassName(raw),
    raw_class: String(raw),
    confidence: Number(pred.confidence ?? pred.score ?? 0),
    x: cx - w / 2,
    y: cy - h / 2,
    width: w,
    height: h,
    model_class_id: pred.class_id ?? pred.classId ?? null,
  };
}

export function filterByConfidence(detections, threshold) {
  const t = Number(threshold);
  if (!Number.isFinite(t)) return detections;
  return detections.filter((d) => d.confidence >= t);
}

export function countByClass(detections) {
  const counts = {};
  for (const d of detections) {
    counts[d.class] = (counts[d.class] || 0) + 1;
  }
  return counts;
}

/**
 * @typedef {object} DetectorAdapter
 * @property {() => object} getModelInfo
 * @property {(jpeg: Buffer, opts?: object) => Promise<{detections: object[], raw: any, timings_ms: object, image_size: object}>} detect
 */

export class BaseDetectorAdapter {
  getModelInfo() {
    throw new Error('getModelInfo() not implemented');
  }

  async detect(_jpeg, _opts = {}) {
    throw new Error('detect() not implemented');
  }
}
