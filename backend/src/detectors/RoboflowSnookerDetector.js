/**
 * Roboflow hosted snooker-ball detector adapter.
 *
 * Model (benchmark target):
 *   snooker-ball-detection-rnhxo-95km5/2
 *
 * Secret API key stays on the backend only (ROBOFLOW_API_KEY).
 * Inference is intentionally low-rate — callers must throttle.
 */

import { BaseDetectorAdapter, fromCenterBox } from './DetectorAdapter.js';

const DEFAULT_MODEL_ID = 'snooker-ball-detection-rnhxo-95km5/2';
const DEFAULT_API_URL = 'https://serverless.roboflow.com';

export class RoboflowSnookerDetector extends BaseDetectorAdapter {
  constructor({
    apiKey = process.env.ROBOFLOW_API_KEY || '',
    modelId = process.env.ROBOFLOW_MODEL_ID || DEFAULT_MODEL_ID,
    apiUrl = process.env.ROBOFLOW_API_URL || DEFAULT_API_URL,
  } = {}) {
    super();
    this.apiKey = String(apiKey || '').trim();
    this.modelId = String(modelId || DEFAULT_MODEL_ID).trim();
    this.apiUrl = String(apiUrl || DEFAULT_API_URL).replace(/\/+$/, '');
    this._classCache = null;
  }

  getModelInfo() {
    return {
      id: 'roboflow-snooker',
      name: 'Roboflow Snooker Ball Detection',
      provider: 'roboflow',
      model_id: this.modelId,
      api_url: this.apiUrl,
      configured: Boolean(this.apiKey),
      requires_api_key: true,
      env_key: 'ROBOFLOW_API_KEY',
      note:
        'Hosted inference. Class names come from the model response — mapped to SCOS names when possible.',
      known_related_classes: [
        'white/cue',
        'red',
        'yellow',
        'green',
        'brown',
        'blue',
        'pink',
        'black',
      ],
    };
  }

  async detect(jpegBuffer, { confidence = 0.01, sourceImageSize = null } = {}) {
    if (!this.apiKey) {
      const err = new Error(
        'ROBOFLOW_API_KEY is not set. Add it to backend/.env (see backend/.env.example).'
      );
      err.status = 503;
      err.code = 'ROBOFLOW_API_KEY_MISSING';
      throw err;
    }
    if (!jpegBuffer?.length) {
      const err = new Error('Empty image for detection');
      err.status = 400;
      throw err;
    }

    const confPct = Math.max(1, Math.min(99, Math.round(Number(confidence) * 100) || 1));
    const url =
      `${this.apiUrl}/${this.modelId}` +
      `?api_key=${encodeURIComponent(this.apiKey)}` +
      `&confidence=${confPct}` +
      `&overlap=30`;

    const t0 = Date.now();
    const res = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: jpegBuffer.toString('base64'),
    });
    const networkMs = Date.now() - t0;
    const raw = await res.json().catch(() => ({}));

    if (!res.ok) {
      const msg =
        raw?.error || raw?.message || raw?.detail || `Roboflow HTTP ${res.status}`;
      const err = new Error(String(msg));
      err.status = res.status === 401 || res.status === 403 ? 502 : 502;
      err.code = 'ROBOFLOW_INFER_FAILED';
      err.raw = raw;
      throw err;
    }

    const preds = Array.isArray(raw.predictions) ? raw.predictions : [];
    const inferW = Number(raw.image?.width) || null;
    const inferH = Number(raw.image?.height) || null;

    let scaleX = 1;
    let scaleY = 1;
    if (sourceImageSize?.width && inferW) scaleX = sourceImageSize.width / inferW;
    if (sourceImageSize?.height && inferH) scaleY = sourceImageSize.height / inferH;

    const detections = preds.map((p) => fromCenterBox(p, { scaleX, scaleY }));

    // Remember class vocabulary seen from this model
    const seen = new Set(detections.map((d) => d.raw_class));
    if (seen.size) {
      this._classCache = [...new Set([...(this._classCache || []), ...seen])].sort();
    }

    return {
      detections,
      raw,
      timings_ms: {
        network: networkMs,
        total: networkMs,
      },
      image_size: {
        inference_width: inferW,
        inference_height: inferH,
        source_width: sourceImageSize?.width || null,
        source_height: sourceImageSize?.height || null,
      },
      model_classes_seen: this._classCache,
    };
  }
}
