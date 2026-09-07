/**
 * Hugging Face Space detector adapter.
 *
 * Calls a SCOS-compatible /detect endpoint hosted on Hugging Face Spaces.
 * The Space loads a YOLOv8 model from HF Hub and serves inference via FastAPI.
 *
 * Detection response is already SCOS-normalized (top-left x,y + width,height).
 * No center-box conversion needed — unlike the Roboflow adapter.
 */

import { BaseDetectorAdapter } from './DetectorAdapter.js';

const DEFAULT_SPACE_URL =
  process.env.HF_SPACE_URL || process.env.HF_DETECTOR_URL || '';

export class HFDetector extends BaseDetectorAdapter {
  constructor({
    spaceUrl = DEFAULT_SPACE_URL,
    confidence = 0.25,
    iou = 0.45,
  } = {}) {
    super();
    this.spaceUrl = String(spaceUrl || '').replace(/\/+$/, '');
    this.defaultConfidence = Number(confidence) || 0.25;
    this.defaultIou = Number(iou) || 0.45;

    if (!this.spaceUrl) {
      console.warn(
        '[HFDetector] No HF_SPACE_URL configured. Set it in backend/.env or pass spaceUrl.'
      );
    }
  }

  getModelInfo() {
    return {
      id: 'hf-space',
      name: 'Hugging Face Space (YOLOv8)',
      provider: 'huggingface',
      space_url: this.spaceUrl,
      configured: Boolean(this.spaceUrl),
      requires_api_key: false,
      env_key: 'HF_SPACE_URL',
      note:
        'Self-hosted YOLOv8 on Hugging Face Spaces (free CPU). ' +
        'No API key needed. Model weights pulled from HF Hub on Space startup.',
      known_classes: [
        'red', 'yellow', 'green', 'brown',
        'blue', 'pink', 'black', 'cue_ball',
      ],
    };
  }

  async detect(jpegBuffer, { confidence, iou, sourceImageSize = null } = {}) {
    if (!this.spaceUrl) {
      const err = new Error(
        'HF_SPACE_URL is not set. Add it to backend/.env (see backend/.env.example).'
      );
      err.status = 503;
      err.code = 'HF_SPACE_URL_MISSING';
      throw err;
    }
    if (!jpegBuffer?.length) {
      const err = new Error('Empty image for detection');
      err.status = 400;
      throw err;
    }

    const conf = confidence != null ? confidence : this.defaultConfidence;
    const iouThr = iou != null ? iou : this.defaultIou;

    const form = new FormData();
    form.append('file', new Blob([jpegBuffer], { type: 'image/jpeg' }), 'frame.jpg');
    form.append('confidence', String(conf));
    form.append('iou', String(iouThr));

    const t0 = Date.now();
    const res = await fetch(`${this.spaceUrl}/detect`, {
      method: 'POST',
      body: form,
    });
    const networkMs = Date.now() - t0;
    const data = await res.json().catch(() => ({}));

    if (!res.ok) {
      const detail = Array.isArray(data.detail)
        ? data.detail.map((d) => d.msg || d).join('; ')
        : data.detail || data.error || `HF Space HTTP ${res.status}`;
      const err = new Error(detail);
      err.status = res.status === 401 || res.status === 403 ? 502 : 502;
      err.code = 'HF_INFER_FAILED';
      err.raw = data;
      throw err;
    }

    const detections = Array.isArray(data.detections) ? data.detections : [];
    const inferW = data.image_size?.width || null;
    const inferH = data.image_size?.height || null;

    // Scale detections if the inference image size differs from source
    let scaled = detections;
    if (sourceImageSize?.width && inferW && inferW !== sourceImageSize.width) {
      const scaleX = sourceImageSize.width / inferW;
      const scaleY = sourceImageSize.height / inferH;
      scaled = detections.map((d) => ({
        ...d,
        x: d.x * scaleX,
        y: d.y * scaleY,
        width: d.width * scaleX,
        height: d.height * scaleY,
      }));
    }

    return {
      detections: scaled,
      raw: data,
      timings_ms: {
        network: networkMs,
        inference: data.timings_ms?.inference ?? null,
        total: networkMs,
      },
      image_size: {
        inference_width: inferW,
        inference_height: inferH,
        source_width: sourceImageSize?.width || null,
        source_height: sourceImageSize?.height || null,
      },
      model_classes_seen: null,
    };
  }
}
