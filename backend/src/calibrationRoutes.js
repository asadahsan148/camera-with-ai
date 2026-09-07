/**
 * SCOS Phase 1 — table calibration + vision diagnostics API.
 *
 * Reads RTSP URLs that the existing camera/ONVIF/Dahua code already produced;
 * it never discovers, authenticates or streams cameras itself.
 */

import express from 'express';
import { FrameSource } from './frameSource.js';
import { TableStore, cornersForFrame, calibrationStatus, DEFAULTS } from './tableStore.js';

const AI_URL = process.env.AI_WORKER_URL || 'http://127.0.0.1:5051';
const FPS_WINDOW = 12;

export function createCalibrationRouter({
  getCamera,
  listCameras = () => [],
  store: injectedStore = null,
  frames: injectedFrames = null,
}) {
  const router = express.Router();
  const store = injectedStore || new TableStore();
  const frames = injectedFrames || new FrameSource({ fps: Number(process.env.CALIB_FPS || 4) });
  const fpsWindows = new Map();
  const ownsFrames = !injectedFrames;

  function trackProcessingFps(key) {
    const now = Date.now();
    const times = fpsWindows.get(key) || [];
    times.push(now);
    while (times.length > FPS_WINDOW) times.shift();
    fpsWindows.set(key, times);
    if (times.length < 2) return null;
    const span = times[times.length - 1] - times[0];
    if (span <= 0) return null;
    return Number((((times.length - 1) * 1000) / span).toFixed(2));
  }

  function resolveCamera(cameraId) {
    const camera = getCamera(cameraId);
    if (!camera) {
      const err = new Error('Camera not found');
      err.status = 404;
      throw err;
    }
    if (!camera.rtspUrl) {
      const err = new Error(
        'Camera has no verified RTSP URL — connect it on the Cameras tab first'
      );
      err.status = 409;
      throw err;
    }
    return camera;
  }

  /** Resolve the live camera for a saved table, rebinding UUID if needed. */
  function resolveTableCamera(table) {
    let current = store.rebindTable(table, listCameras());
    if (!current) {
      const err = new Error('Table not found');
      err.status = 404;
      throw err;
    }
    const camera = getCamera(current.camera_id);
    if (!camera?.rtspUrl) {
      const err = new Error(
        'Saved calibration found, but its camera is not connected yet — ' +
          'Load all channels / Connect on the Cameras tab, then reopen this table.'
      );
      err.status = 409;
      throw err;
    }
    return { table: current, camera };
  }

  function publicCamera(camera) {
    return {
      id: camera.id,
      name: camera.name,
      ip: camera.ip,
      channel: camera.channel ?? null,
      nvrIp: camera.nvrIp || null,
    };
  }

  function withStatus(table) {
    return { ...table, calibration_status: calibrationStatus(table) };
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
      err.status = res.status === 400 ? 400 : 502;
      throw err;
    }
    return data;
  }

  async function workerMatrix(body) {
    const res = await fetch(`${AI_URL}/calibration/matrix`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
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

  // --- table configs -------------------------------------------------------

  router.get('/tables', (req, res) => {
    const cameraId = req.query.cameraId;
    if (cameraId) {
      const camera = getCamera(cameraId);
      if (!camera) {
        return res.status(404).json({ error: 'Camera not found', tables: [], defaults: DEFAULTS });
      }
      const tables = store.listForCamera(camera);
      return res.json({ tables: tables.map(withStatus), defaults: DEFAULTS });
    }
    const tables = store.list();
    res.json({ tables: tables.map(withStatus), defaults: DEFAULTS });
  });

  router.post('/tables', (req, res) => {
    try {
      const camera = getCamera(req.body?.camera_id);
      if (!camera) {
        return res.status(404).json({ error: 'Camera not found — connect it first' });
      }
      const table = store.create({
        ...req.body,
        camera,
        camera_name: req.body?.camera_name || camera.name || null,
      });
      res.status(201).json({ table: withStatus(table) });
    } catch (err) {
      res.status(400).json({ error: err.message });
    }
  });

  router.get('/tables/:tableId', (req, res) => {
    let table = store.get(req.params.tableId);
    if (!table) return res.status(404).json({ error: 'Table not found' });
    table = store.rebindTable(table, listCameras()) || table;
    res.json({ table: withStatus(table) });
  });

  router.patch('/tables/:tableId', (req, res) => {
    const allowed = ['name', 'surface_width', 'aspect', 'margin', 'camera_name'];
    const patch = {};
    for (const key of allowed) {
      if (req.body?.[key] !== undefined) patch[key] = req.body[key];
    }
    const table = store.update(req.params.tableId, patch);
    if (!table) return res.status(404).json({ error: 'Table not found' });
    res.json({ table: withStatus(table) });
  });

  router.delete('/tables/:tableId', (req, res) => {
    const ok = store.remove(req.params.tableId);
    if (!ok) return res.status(404).json({ error: 'Table not found' });
    res.json({ ok: true });
  });

  router.post('/tables/:tableId/reset', (req, res) => {
    const table = store.resetCalibration(req.params.tableId);
    if (!table) return res.status(404).json({ error: 'Table not found' });
    res.json({ table: withStatus(table) });
  });

  // --- calibration ---------------------------------------------------------

  router.post('/tables/:tableId/calibrate', async (req, res) => {
    const table = store.get(req.params.tableId);
    if (!table) return res.status(404).json({ error: 'Table not found' });

    const {
      corners,
      reference_width: refW,
      reference_height: refH,
      surface_width: surfaceWidth,
      aspect,
      margin,
    } = req.body || {};

    if (!Array.isArray(corners) || corners.length !== 4) {
      return res.status(400).json({
        error: 'Exactly 4 corners required, in order: top-left, top-right, bottom-right, bottom-left',
      });
    }
    if (!refW || !refH) {
      return res.status(400).json({ error: 'reference_width and reference_height are required' });
    }

    try {
      const camera = getCamera(table.camera_id) || getCamera(req.body?.camera_id);
      const result = await workerMatrix({
        corners,
        frame_width: refW,
        frame_height: refH,
        surface_width: surfaceWidth ?? table.surface_width,
        aspect: aspect ?? table.aspect,
        margin: margin ?? table.margin,
      });

      if (!result.ok) {
        return res.status(400).json({ error: result.errors?.join(' ') || 'Invalid corners' });
      }

      const saved = store.saveCalibration(req.params.tableId, {
        corners: corners.map(([x, y]) => [Number(x), Number(y)]),
        referenceWidth: Number(refW),
        referenceHeight: Number(refH),
        matrix: result.matrix,
        output: result.output,
        camera: camera || undefined,
      });

      res.json({
        table: withStatus(saved),
        quad: result.quad,
        warnings: result.warnings || [],
      });
    } catch (err) {
      res.status(err.status || 502).json({
        error: err.message,
        tip: 'AI worker (port 5051) must be running: npm run dev:ai',
      });
    }
  });

  // --- live frames ---------------------------------------------------------

  router.get('/cameras/:cameraId/snapshot', async (req, res) => {
    try {
      const camera = resolveCamera(req.params.cameraId);
      const started = Date.now();
      const frame = await frames.grab(camera.id, camera.rtspUrl);
      res.json({
        camera: publicCamera(camera),
        image: `data:image/jpeg;base64,${frame.jpeg.toString('base64')}`,
        width: frame.width,
        height: frame.height,
        resolution: frame.width && frame.height ? `${frame.width}x${frame.height}` : null,
        camera_fps: frame.sourceFps,
        capture_fps: frame.captureFps,
        codec: frame.codec,
        grab_ms: Date.now() - started,
        captured_at: new Date(frame.capturedAt).toISOString(),
      });
    } catch (err) {
      res.status(err.status || 502).json({ error: err.message });
    }
  });

  router.post('/cameras/:cameraId/release', (req, res) => {
    res.json({ ok: frames.release(req.params.cameraId) });
  });

  // --- diagnostics ---------------------------------------------------------

  router.get('/tables/:tableId/diagnostics', async (req, res) => {
    const existing = store.get(req.params.tableId);
    if (!existing) return res.status(404).json({ error: 'Table not found' });

    const wantOverlay = req.query.overlay !== 'false';

    try {
      const { table, camera } = resolveTableCamera(existing);
      const status = calibrationStatus(table);
      const t0 = Date.now();
      const frame = await frames.grab(camera.id, camera.rtspUrl);
      const grabMs = Date.now() - t0;

      const base = {
        table: withStatus(table),
        camera: publicCamera(camera),
        source: {
          resolution: frame.width && frame.height ? `${frame.width}x${frame.height}` : null,
          width: frame.width,
          height: frame.height,
          camera_fps: frame.sourceFps,
          capture_fps: frame.captureFps,
          codec: frame.codec,
          frame_age_ms: frame.ageMs,
        },
        raw_image: `data:image/jpeg;base64,${frame.jpeg.toString('base64')}`,
        warped_image: null,
        overlay_image: null,
        timings_ms: { grab: grabMs, transform: null, worker_total: null, request: null },
        processing_fps: trackProcessingFps(table.table_id),
        warnings: [],
      };

      if (status !== 'calibrated') {
        base.timings_ms.request = Date.now() - t0;
        return res.json({ ...base, note: 'Table not calibrated yet — click 4 corners on the left.' });
      }

      const corners = cornersForFrame(table, frame.width, frame.height);
      if (!corners) {
        base.timings_ms.request = Date.now() - t0;
        return res.json({ ...base, note: 'Frame size unknown — could not scale saved corners.' });
      }

      const warp = await workerWarp(frame.jpeg, {
        corners,
        surfaceWidth: table.surface_width,
        aspect: table.aspect,
        margin: table.margin,
        overlay: wantOverlay,
        label: table.name,
      });

      base.warped_image = `data:image/jpeg;base64,${warp.warped_jpeg_base64}`;
      if (warp.overlay_jpeg_base64) {
        base.overlay_image = `data:image/jpeg;base64,${warp.overlay_jpeg_base64}`;
      }
      base.timings_ms.transform = warp.timings_ms?.transform ?? null;
      base.timings_ms.worker_total = warp.timings_ms?.total ?? null;
      base.timings_ms.request = Date.now() - t0;
      base.output = warp.output;
      base.quad = warp.quad;
      base.warnings = warp.warnings || [];
      base.corners_in_frame = corners.map(([x, y]) => [Number(x.toFixed(1)), Number(y.toFixed(1))]);

      res.json(base);
    } catch (err) {
      res.status(err.status || 502).json({ error: err.message });
    }
  });

  /** Preview a warp for corners that are not saved yet. */
  router.post('/tables/:tableId/preview', async (req, res) => {
    const existing = store.get(req.params.tableId);
    if (!existing) return res.status(404).json({ error: 'Table not found' });

    const { corners, reference_width: refW, reference_height: refH } = req.body || {};
    if (!Array.isArray(corners) || corners.length !== 4) {
      return res.status(400).json({ error: 'Exactly 4 corners required' });
    }

    try {
      const { table, camera } = resolveTableCamera(existing);
      const frame = await frames.grab(camera.id, camera.rtspUrl);
      const scaled =
        refW && refH && frame.width && frame.height
          ? corners.map(([x, y]) => [(x / refW) * frame.width, (y / refH) * frame.height])
          : corners;

      const warp = await workerWarp(frame.jpeg, {
        corners: scaled,
        surfaceWidth: req.body.surface_width ?? table.surface_width,
        aspect: req.body.aspect ?? table.aspect,
        margin: req.body.margin ?? table.margin,
        overlay: false,
        label: table.name,
      });

      res.json({
        warped_image: `data:image/jpeg;base64,${warp.warped_jpeg_base64}`,
        output: warp.output,
        quad: warp.quad,
        warnings: warp.warnings || [],
        timings_ms: warp.timings_ms,
      });
    } catch (err) {
      res.status(err.status || 502).json({ error: err.message });
    }
  });

  router.stopAll = () => {
    if (ownsFrames) frames.stopAll();
  };
  return router;
}
