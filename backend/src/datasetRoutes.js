/**
 * SCOS Phase 2 — vision dataset capture & annotation API.
 *
 * Reuses Phase 1 TableStore + FrameSource + AI /calibration/warp.
 * Does not touch RTSP discovery/auth or the live mpegts StreamManager.
 */

import express from 'express';
import path from 'path';
import { FrameSource } from './frameSource.js';
import { TableStore, cornersForFrame, calibrationStatus } from './tableStore.js';
import { DatasetStore, DATASET_CLASSES, DEFAULT_DATASET_ROOT } from './datasetStore.js';

const AI_URL = process.env.AI_WORKER_URL || 'http://127.0.0.1:5051';

export function createDatasetRouter({
  getCamera,
  listCameras = () => [],
  store: injectedStore = null,
  frames: injectedFrames = null,
  datasetRoot = DEFAULT_DATASET_ROOT,
}) {
  const router = express.Router();
  const store = injectedStore || new TableStore();
  const frames = injectedFrames || new FrameSource({ fps: Number(process.env.CALIB_FPS || 4) });
  const dataset = new DatasetStore(datasetRoot);
  const ownsFrames = !injectedFrames;

  function resolveCamera(cameraId) {
    const camera = getCamera(cameraId);
    if (!camera?.rtspUrl) {
      const err = new Error('Camera not connected — load channels / Connect first');
      err.status = 409;
      throw err;
    }
    return camera;
  }

  function resolveTable(tableId) {
    let table = store.get(tableId);
    if (!table) {
      const err = new Error('Table not found');
      err.status = 404;
      throw err;
    }
    table = store.rebindTable(table, listCameras()) || table;
    if (calibrationStatus(table) !== 'calibrated') {
      const err = new Error('Table is not calibrated — finish Phase 1 calibration first');
      err.status = 409;
      throw err;
    }
    const camera = resolveCamera(table.camera_id);
    return { table, camera };
  }

  async function workerWarp(jpeg, { corners, surfaceWidth, aspect, margin, overlay, label }) {
    const form = new FormData();
    form.append('file', new Blob([jpeg], { type: 'image/jpeg' }), 'frame.jpg');
    form.append('corners', JSON.stringify(corners));
    form.append('surface_width', String(surfaceWidth));
    form.append('aspect', String(aspect));
    form.append('margin', String(margin));
    form.append('draw_overlay', overlay ? 'true' : 'false');
    if (label) form.append('table_label', label);

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

  async function grabWarped(tableId, { overlay = true } = {}) {
    const { table, camera } = resolveTable(tableId);
    const frame = await frames.grab(camera.id, camera.rtspUrl);
    const corners = cornersForFrame(table, frame.width, frame.height);
    if (!corners) {
      const err = new Error('Could not scale calibration corners to current frame');
      err.status = 500;
      throw err;
    }
    const warp = await workerWarp(frame.jpeg, {
      corners,
      surfaceWidth: table.surface_width,
      aspect: table.aspect,
      margin: table.margin,
      overlay,
      label: table.name,
    });
    return { table, camera, frame, corners, warp };
  }

  // --- catalog / stats -----------------------------------------------------

  router.get('/dataset/stats', (_req, res) => {
    res.json(dataset.stats());
  });

  router.get('/dataset/classes', (_req, res) => {
    res.json({ classes: DATASET_CLASSES, ...dataset.classes() });
  });

  router.get('/dataset/images', (req, res) => {
    const annotated =
      req.query.annotated === 'true' ? true : req.query.annotated === 'false' ? false : undefined;
    res.json(
      dataset.list({
        annotated,
        limit: req.query.limit,
        offset: req.query.offset,
      })
    );
  });

  router.get('/dataset/images/:imageId', (req, res) => {
    const row = dataset.get(req.params.imageId);
    if (!row) return res.status(404).json({ error: 'Image not found' });
    res.json({
      image: row,
      url: `/api/dataset/images/${row.image_id}/file`,
    });
  });

  router.get('/dataset/images/:imageId/file', (req, res) => {
    const buf = dataset.readImage(req.params.imageId);
    if (!buf) return res.status(404).json({ error: 'Image file missing' });
    res.setHeader('Content-Type', 'image/jpeg');
    res.setHeader('Cache-Control', 'public, max-age=3600');
    res.send(buf);
  });

  router.delete('/dataset/images/:imageId', (req, res) => {
    const ok = dataset.remove(req.params.imageId);
    if (!ok) return res.status(404).json({ error: 'Image not found' });
    res.json({ ok: true });
  });

  // --- live preview (reuse Phase 1 warp) -----------------------------------

  router.get('/dataset/preview/:tableId', async (req, res) => {
    try {
      const { table, camera, frame, warp } = await grabWarped(req.params.tableId, {
        overlay: req.query.overlay !== 'false',
      });
      res.json({
        table: { ...table, calibration_status: calibrationStatus(table) },
        camera: {
          id: camera.id,
          name: camera.name,
          ip: camera.ip,
          channel: camera.channel ?? null,
        },
        source: {
          width: frame.width,
          height: frame.height,
          resolution: frame.width && frame.height ? `${frame.width}x${frame.height}` : null,
          capture_fps: frame.captureFps,
          camera_fps: frame.sourceFps,
        },
        raw_image: warp.overlay_jpeg_base64
          ? `data:image/jpeg;base64,${warp.overlay_jpeg_base64}`
          : `data:image/jpeg;base64,${frame.jpeg.toString('base64')}`,
        warped_image: `data:image/jpeg;base64,${warp.warped_jpeg_base64}`,
        output: warp.output,
        timings_ms: warp.timings_ms,
      });
    } catch (err) {
      res.status(err.status || 502).json({ error: err.message });
    }
  });

  // --- capture -------------------------------------------------------------

  router.post('/dataset/capture', async (req, res) => {
    const tableId = req.body?.table_id;
    if (!tableId) return res.status(400).json({ error: 'table_id required' });
    const force = Boolean(req.body?.force);
    const similarityThreshold = Number(req.body?.similarity_threshold ?? 8);

    try {
      const { table, camera, frame, warp } = await grabWarped(tableId, { overlay: false });
      const jpeg = Buffer.from(warp.warped_jpeg_base64, 'base64');
      const result = await dataset.addCapture({
        jpeg,
        camera,
        table,
        sourceSize: { width: frame.width, height: frame.height },
        outputSize: {
          width: warp.output?.output_width || table.output_width,
          height: warp.output?.output_height || table.output_height,
        },
        timingsMs: warp.timings_ms,
        force,
        similarityThreshold,
      });

      if (result.skipped) {
        return res.status(200).json({
          ok: true,
          skipped: true,
          reason: result.reason,
          nearest: result.nearest,
          distance: result.distance,
          stats: dataset.stats(),
        });
      }

      res.status(201).json({
        ok: true,
        skipped: false,
        image: result.saved,
        meta: result.meta,
        url: `/api/dataset/images/${result.saved.image_id}/file`,
        stats: dataset.stats(),
      });
    } catch (err) {
      res.status(err.status || 502).json({ error: err.message });
    }
  });

  // --- annotations ---------------------------------------------------------

  router.get('/dataset/images/:imageId/annotation', (req, res) => {
    const row = dataset.get(req.params.imageId);
    if (!row) return res.status(404).json({ error: 'Image not found' });
    res.json({ annotation: row.annotation, classes: DATASET_CLASSES });
  });

  router.put('/dataset/images/:imageId/annotation', (req, res) => {
    try {
      const ann = dataset.saveAnnotation(req.params.imageId, {
        boxes: req.body?.boxes || [],
        width: req.body?.width,
        height: req.body?.height,
      });
      if (!ann) return res.status(404).json({ error: 'Image not found' });
      res.json({ annotation: ann, stats: dataset.stats() });
    } catch (err) {
      res.status(err.status || 400).json({ error: err.message });
    }
  });

  // --- export --------------------------------------------------------------

  router.post('/dataset/export', async (_req, res) => {
    try {
      const result = await dataset.exportPackage();
      res.json({
        ok: true,
        ...result,
        folder_name: path.basename(result.folder),
        zip_name: result.zip ? path.basename(result.zip) : null,
        tip: 'Package is under datasets/scos-v1/exports/ — YOLO labels + JSON boxes, no train/val split yet.',
      });
    } catch (err) {
      res.status(500).json({ error: err.message });
    }
  });

  router.stopAll = () => {
    if (ownsFrames) frames.stopAll();
  };

  return router;
}
