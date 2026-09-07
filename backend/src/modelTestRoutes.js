/**
 * SCOS Phase 2B — pretrained model benchmark API.
 *
 * Pipeline: calibrated table → perspective warp → DetectorAdapter → overlay
 * Isolated from legacy HSV detector and dataset annotation workflow.
 */

import express from 'express';
import { FrameSource } from './frameSource.js';
import { TableStore, cornersForFrame, calibrationStatus } from './tableStore.js';
import { getDetector, listDetectors } from './detectors/index.js';
import {
  filterByConfidence,
  countByClass,
} from './detectors/DetectorAdapter.js';
import { ModelTestSnapshotStore } from './modelTestSnapshots.js';

const AI_URL = process.env.AI_WORKER_URL || 'http://127.0.0.1:5051';

export function createModelTestRouter({
  getCamera,
  listCameras = () => [],
  store: injectedStore = null,
  frames: injectedFrames = null,
}) {
  const router = express.Router();
  const store = injectedStore || new TableStore();
  const frames = injectedFrames || new FrameSource({ fps: Number(process.env.CALIB_FPS || 4) });
  const snapshots = new ModelTestSnapshotStore();
  const ownsFrames = !injectedFrames;

  function resolveTable(tableId) {
    let table = store.get(tableId);
    if (!table) {
      const err = new Error('Table not found');
      err.status = 404;
      throw err;
    }
    table = store.rebindTable(table, listCameras()) || table;
    if (calibrationStatus(table) !== 'calibrated') {
      const err = new Error('Table is not calibrated — finish Phase 1 first');
      err.status = 409;
      throw err;
    }
    const camera = getCamera(table.camera_id);
    if (!camera?.rtspUrl) {
      const err = new Error('Camera not connected — load channels first');
      err.status = 409;
      throw err;
    }
    return { table, camera };
  }

  async function workerWarp(jpeg, opts) {
    const form = new FormData();
    form.append('file', new Blob([jpeg], { type: 'image/jpeg' }), 'frame.jpg');
    form.append('corners', JSON.stringify(opts.corners));
    form.append('surface_width', String(opts.surfaceWidth));
    form.append('aspect', String(opts.aspect));
    form.append('margin', String(opts.margin));
    form.append('draw_overlay', 'false');
    if (opts.label) form.append('table_label', opts.label);

    const res = await fetch(`${AI_URL}/calibration/warp`, { method: 'POST', body: form });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const detail = Array.isArray(data.detail)
        ? data.detail.map((d) => d.msg || d).join('; ')
        : data.detail || data.error;
      const err = new Error(detail || `AI worker HTTP ${res.status}`);
      err.status = 502;
      throw err;
    }
    return data;
  }

  async function workerDrawBoxes(jpeg, detections, threshold) {
    const form = new FormData();
    form.append('file', new Blob([jpeg], { type: 'image/jpeg' }), 'frame.jpg');
    form.append('detections', JSON.stringify(detections));
    form.append('confidence_threshold', String(threshold));

    const res = await fetch(`${AI_URL}/vision/draw_boxes`, { method: 'POST', body: form });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const detail = Array.isArray(data.detail)
        ? data.detail.map((d) => d.msg || d).join('; ')
        : data.detail || data.error || `draw_boxes HTTP ${res.status}`;
      const err = new Error(detail);
      err.status = 502;
      throw err;
    }
    return data;
  }

  async function grabWarped(tableId) {
    const { table, camera } = resolveTable(tableId);
    const frame = await frames.grab(camera.id, camera.rtspUrl);
    const corners = cornersForFrame(table, frame.width, frame.height);
    if (!corners) {
      const err = new Error('Could not scale calibration corners to frame');
      err.status = 500;
      throw err;
    }
    const warp = await workerWarp(frame.jpeg, {
      corners,
      surfaceWidth: table.surface_width,
      aspect: table.aspect,
      margin: table.margin,
      label: table.name,
    });
    const warpedJpeg = Buffer.from(warp.warped_jpeg_base64, 'base64');
    return {
      table,
      camera,
      frame,
      warp,
      warpedJpeg,
      outputSize: {
        width: warp.output?.output_width || table.output_width,
        height: warp.output?.output_height || table.output_height,
      },
    };
  }

  router.get('/model-test/detectors', (_req, res) => {
    res.json({ detectors: listDetectors() });
  });

  router.get('/model-test/status', (_req, res) => {
    const detectors = listDetectors();
    const primary = detectors.find((d) => d.id === 'roboflow-snooker') || detectors[0];
    res.json({
      ok: true,
      detector: primary,
      api_key_configured: Boolean(primary?.configured),
      tip: primary?.configured
        ? null
        : 'Set ROBOFLOW_API_KEY in backend/.env then restart the backend.',
    });
  });

  /**
   * One benchmark tick:
   * grab calibrated warp → run detector → filter by confidence → optional overlay
   */
  router.post('/model-test/infer', async (req, res) => {
    const tableId = req.body?.table_id;
    const detectorId = req.body?.detector_id || 'roboflow-snooker';
    const confidence = Number(req.body?.confidence ?? 0.4);

    if (!tableId) return res.status(400).json({ error: 'table_id required' });

    try {
      const t0 = Date.now();
      const { table, camera, frame, warp, warpedJpeg, outputSize } = await grabWarped(tableId);
      const grabMs = Date.now() - t0;

      const detector = getDetector(detectorId);
      const modelInfo = detector.getModelInfo();
      const result = await detector.detect(warpedJpeg, {
        confidence: 0.01, // fetch low; filter client/server-side for threshold UI
        sourceImageSize: outputSize,
      });

      const all = result.detections || [];
      const filtered = filterByConfidence(all, confidence);
      const low = all.filter((d) => d.confidence < 0.5);

      const inferMs = result.timings_ms?.total ?? null;
      const requestMs = Date.now() - t0;

      res.json({
        ok: true,
        model: modelInfo,
        table: {
          table_id: table.table_id,
          name: table.name,
          camera_id: table.camera_id,
          calibration_timestamp: table.calibration_timestamp,
          calibration_status: calibrationStatus(table),
        },
        camera: { id: camera.id, name: camera.name, channel: camera.channel ?? null },
        source: {
          width: frame.width,
          height: frame.height,
          warped_width: outputSize.width,
          warped_height: outputSize.height,
        },
        clean_image: `data:image/jpeg;base64,${warp.warped_jpeg_base64}`,
        annotated_image: null,
        detections: filtered.map((d) => ({
          ...d,
          x: Math.round(d.x * 10) / 10,
          y: Math.round(d.y * 10) / 10,
          width: Math.round(d.width * 10) / 10,
          height: Math.round(d.height * 10) / 10,
          confidence: Math.round(d.confidence * 1000) / 1000,
        })),
        detections_all: all.map((d) => ({
          ...d,
          x: Math.round(d.x * 10) / 10,
          y: Math.round(d.y * 10) / 10,
          width: Math.round(d.width * 10) / 10,
          height: Math.round(d.height * 10) / 10,
          confidence: Math.round(d.confidence * 1000) / 1000,
        })),
        stats: {
          confidence_threshold: confidence,
          total_detections: filtered.length,
          total_raw: all.length,
          low_confidence_count: low.length,
          counts_by_class: countByClass(filtered),
          model_classes_seen: result.model_classes_seen || [],
        },
        timings_ms: {
          grab_and_warp: grabMs,
          inference: inferMs,
          draw: null,
          request: requestMs,
          roboflow: result.timings_ms || null,
          warp: warp.timings_ms || null,
        },
        raw_prediction_summary: {
          prediction_count: Array.isArray(result.raw?.predictions)
            ? result.raw.predictions.length
            : 0,
          image: result.raw?.image || null,
        },
      });
    } catch (err) {
      res.status(err.status || 502).json({
        error: err.message,
        code: err.code || null,
        tip:
          err.code === 'ROBOFLOW_API_KEY_MISSING'
            ? 'Create backend/.env with ROBOFLOW_API_KEY=... then restart backend.'
            : null,
      });
    }
  });

  router.post('/model-test/snapshot', async (req, res) => {
    const tableId = req.body?.table_id;
    const detectorId = req.body?.detector_id || 'roboflow-snooker';
    const confidence = Number(req.body?.confidence ?? 0.4);
    if (!tableId) return res.status(400).json({ error: 'table_id required' });

    try {
      const { table, camera, warpedJpeg, outputSize, warp, frame } = await grabWarped(tableId);
      const detector = getDetector(detectorId);
      const modelInfo = detector.getModelInfo();
      const result = await detector.detect(warpedJpeg, {
        confidence: 0.01,
        sourceImageSize: outputSize,
      });
      const filtered = filterByConfidence(result.detections || [], confidence);
      const drawn = await workerDrawBoxes(warpedJpeg, filtered, confidence);
      const annotatedJpeg = Buffer.from(drawn.image_jpeg_base64, 'base64');

      const saved = snapshots.save({
        cleanJpeg: warpedJpeg,
        annotatedJpeg,
        predictions: filtered,
        rawPredictions: result.raw || null,
        confidenceThreshold: confidence,
        modelInfo,
        camera,
        table,
        timings: result.timings_ms,
        imageSize: {
          source: { width: frame.width, height: frame.height },
          warped: outputSize,
          inference: result.image_size,
        },
      });

      // Also keep raw model payload for later comparison
      res.json({
        ok: true,
        snapshot: saved,
        tip: 'Saved under datasets/model-tests/',
      });
    } catch (err) {
      res.status(err.status || 502).json({ error: err.message, code: err.code || null });
    }
  });

  router.get('/model-test/snapshots', (req, res) => {
    res.json(snapshots.list(Number(req.query.limit) || 50));
  });

  router.stopAll = () => {
    if (ownsFrames) frames.stopAll();
  };

  return router;
}
