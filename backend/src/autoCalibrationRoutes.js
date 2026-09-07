/**
 * SCOS Auto-Calibration — AI-powered table corner detection.
 *
 * Uses a YOLOv8-pose model hosted on Hugging Face Spaces to automatically
 * detect the 4 snooker table corners (TL, TR, BR, BL) from a live camera frame.
 *
 * Replaces manual click-to-calibrate with a single API call.
 * After auto-detecting corners, the result is fed into the existing
 * perspective matrix pipeline (same as manual calibration).
 *
 * Routes:
 *   POST /auto-calibrate/:cameraId  — grab frame, detect corners, return result
 *   POST /auto-calibrate/:cameraId/save — detect + save as calibration
 *   GET  /auto-calibrate/status     — check if HF Space is reachable
 */

import express from 'express';
import { FrameSource } from './frameSource.js';
import { TableStore } from './tableStore.js';

const AI_URL       = process.env.AI_WORKER_URL        || 'http://127.0.0.1:5051';
const HF_SPACE_URL = (process.env.HF_CORNER_SPACE_URL || '').replace(/\/+$/, '');

export function createAutoCalibrationRouter({
  getCamera,
  store: injectedStore   = null,
  frames: injectedFrames = null,
}) {
  const router = express.Router();
  const store  = injectedStore  || new TableStore();
  const frames = injectedFrames || new FrameSource({ fps: 2 });

  function resolveCamera(cameraId) {
    const cam = getCamera(cameraId);
    if (!cam) {
      const err = new Error(`Camera not found: ${cameraId}`);
      err.status = 404;
      throw err;
    }
    if (!cam.rtspUrl) {
      const err = new Error(`Camera ${cameraId} has no RTSP URL`);
      err.status = 409;
      throw err;
    }
    return cam;
  }

  /** Grab a frame from the camera via FrameSource. */
  async function grabFrame(camera) {
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error('Frame grab timeout')), 15000);
      frames.getFrame(camera.rtspUrl, (err, jpeg) => {
        clearTimeout(timeout);
        if (err) return reject(err);
        resolve(jpeg);
      });
    });
  }

  /** Call the HF Space /detect-corners endpoint. */
  async function detectCornersHF(jpegBuffer, confidence = 0.25) {
    if (!HF_SPACE_URL) {
      const err = new Error(
        'HF_CORNER_SPACE_URL is not set. Add it to backend/.env (see .env.example).'
      );
      err.status = 503;
      err.code   = 'HF_CORNER_SPACE_URL_MISSING';
      throw err;
    }

    const form = new FormData();
    form.append('file', new Blob([jpegBuffer], { type: 'image/jpeg' }), 'frame.jpg');
    form.append('confidence', String(confidence));

    const res = await fetch(`${HF_SPACE_URL}/detect-corners`, { method: 'POST', body: form });
    const data = await res.json().catch(() => ({}));

    if (!res.ok) {
      const msg = data?.detail || data?.error || `HF Space HTTP ${res.status}`;
      const err = new Error(String(msg));
      err.status = 502;
      err.code   = 'HF_CORNER_INFER_FAILED';
      throw err;
    }

    return data;
  }

  /** Call AI worker /calibration/matrix to compute perspective matrix from corners. */
  async function computeMatrix(corners, imgWidth, imgHeight, tableOptions = {}) {
    const body = {
      corners,            // [{x, y}, {x, y}, {x, y}, {x, y}] — TL, TR, BR, BL
      image_width:  imgWidth,
      image_height: imgHeight,
      surface_width:  tableOptions.surface_width  || 3569,
      aspect:         tableOptions.aspect         || 0.5,
      margin:         tableOptions.margin         || 0.08,
      output_width:   tableOptions.output_width   || 960,
    };

    const res = await fetch(`${AI_URL}/calibration/matrix`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(body),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const err = new Error(data?.detail || `AI worker HTTP ${res.status}`);
      err.status = 502;
      throw err;
    }
    return data;
  }

  // ─── GET /auto-calibrate/status ────────────────────────────────────────────

  router.get('/status', async (req, res) => {
    const status = {
      hf_space_configured: Boolean(HF_SPACE_URL),
      hf_space_url:        HF_SPACE_URL || null,
      ai_worker_url:       AI_URL,
    };

    if (!HF_SPACE_URL) {
      return res.json({ ...status, hf_space_reachable: false });
    }

    try {
      const r = await fetch(`${HF_SPACE_URL}/health`, { signal: AbortSignal.timeout(5000) });
      const d = await r.json().catch(() => ({}));
      status.hf_space_reachable = r.ok;
      status.hf_space_health    = d;
    } catch (err) {
      status.hf_space_reachable = false;
      status.hf_space_error     = err.message;
    }

    res.json(status);
  });

  // ─── POST /auto-calibrate/:cameraId ────────────────────────────────────────

  router.post('/:cameraId', async (req, res) => {
    try {
      const camera = resolveCamera(req.params.cameraId);
      const confidence = Number(req.body?.confidence || req.query?.confidence || 0.3);

      const jpeg = await grabFrame(camera);
      const detection = await detectCornersHF(jpeg, confidence);

      if (!detection.found) {
        return res.status(422).json({
          ok:      false,
          error:   'Table corners not detected in frame. Try adjusting camera angle.',
          code:    'CORNERS_NOT_FOUND',
          confidence_used: confidence,
          image_size: detection.image_size,
        });
      }

      // Map HF response to SCOS corner format: [{x, y}, ...]
      const corners = detection.corners.map((c) => ({ x: c.x, y: c.y }));

      res.json({
        ok:           true,
        corners,
        corner_names: detection.corners.map((c) => c.name),
        confidence:   detection.confidence,
        image_size:   detection.image_size,
        timings_ms:   detection.timings_ms,
        jpeg_b64:     jpeg.toString('base64'),
      });
    } catch (err) {
      res.status(err.status || 500).json({ ok: false, error: err.message, code: err.code });
    }
  });

  // ─── POST /auto-calibrate/:cameraId/save ───────────────────────────────────

  router.post('/:cameraId/save', async (req, res) => {
    try {
      const camera     = resolveCamera(req.params.cameraId);
      const confidence = Number(req.body?.confidence || 0.3);
      const tableName  = req.body?.name || `Auto-${camera.name || camera.id}`;
      const tableOpts  = req.body?.table_options || {};

      const jpeg = await grabFrame(camera);
      const detection = await detectCornersHF(jpeg, confidence);

      if (!detection.found) {
        return res.status(422).json({
          ok:    false,
          error: 'Table corners not detected. Cannot auto-calibrate.',
          code:  'CORNERS_NOT_FOUND',
        });
      }

      const { width: imgW, height: imgH } = detection.image_size;
      const corners = detection.corners.map((c) => ({ x: c.x, y: c.y }));

      // Compute perspective matrix via existing AI worker
      const matResult = await computeMatrix(corners, imgW, imgH, tableOpts);

      if (!matResult.valid) {
        return res.status(422).json({
          ok:    false,
          error: `Calibration invalid: ${matResult.reason || 'degenerate corners'}`,
          code:  'CALIBRATION_INVALID',
          corners,
          matrix_result: matResult,
        });
      }

      // Save using existing TableStore
      const table = store.createOrUpdateTable({
        camera_id:       camera.id,
        camera_key:      camera.nvrIp ? `${camera.nvrIp}|ch${camera.channel || 0}` : camera.id,
        camera_name:     camera.name || null,
        name:            tableName,
        corners,
        matrix:          matResult.matrix,
        output_width:    matResult.output_width   || tableOpts.output_width  || 960,
        output_height:   matResult.output_height  || null,
        surface_width:   tableOpts.surface_width  || 3569,
        aspect:          tableOpts.aspect         || 0.5,
        margin:          tableOpts.margin         || 0.08,
        auto_calibrated: true,
        auto_confidence: detection.confidence,
      });

      res.json({
        ok:              true,
        table,
        corners,
        confidence:      detection.confidence,
        image_size:      detection.image_size,
        matrix_valid:    matResult.valid,
        auto_calibrated: true,
      });
    } catch (err) {
      res.status(err.status || 500).json({ ok: false, error: err.message, code: err.code });
    }
  });

  return router;
}
