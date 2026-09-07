/**
 * SCOS Frame State Routes — REST API for frame start/end detection.
 *
 * Routes:
 *   POST   /frame-state/:tableId/enable   — Start monitoring a table
 *   POST   /frame-state/:tableId/disable  — Stop monitoring
 *   GET    /frame-state/:tableId           — Get current state
 *   GET    /frame-state                    — List all active monitors
 *   GET    /frame-state/events             — Get recent events
 *   DELETE /frame-state/events             — Clear events
 */

import express from 'express';
import { FrameStateManager } from './frameStateManager.js';
import { TableStore, calibrationStatus } from './tableStore.js';

const DEFAULT_SAMPLE_MS = Number(process.env.FRAME_STATE_SAMPLE_MS || 5000);
const DEFAULT_CONFIDENCE = Number(process.env.FRAME_STATE_CONFIDENCE || 0.4);
const DEFAULT_CONFIRM_COUNT = Number(process.env.FRAME_STATE_CONFIRM_COUNT || 3);
const DEFAULT_DETECTOR_ID = process.env.FRAME_STATE_DETECTOR || 'roboflow-snooker';

export function createFrameStateRouter({
  getCamera,
  listCameras = () => [],
  store: injectedStore = null,
  frames: injectedFrames = null,
  manager: injectedManager = null,
}) {
  const router = express.Router();
  const store = injectedStore || new TableStore();
  const frames = injectedFrames;
  const manager = injectedManager || new FrameStateManager();

  function resolveTable(tableId) {
    let table = store.get(tableId);
    if (!table) {
      const err = new Error('Table not found');
      err.status = 404;
      throw err;
    }
    table = store.rebindTable(table, listCameras()) || table;
    if (calibrationStatus(table) !== 'calibrated') {
      const err = new Error('Table is not calibrated — finish calibration first');
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

  // ─── POST /frame-state/:tableId/enable ────────────────────────────

  router.post('/:tableId/enable', async (req, res) => {
    try {
      const { table, camera } = resolveTable(req.params.tableId);

      const detectorId = req.body?.detector_id || DEFAULT_DETECTOR_ID;
      const confidence = Number(req.body?.confidence ?? DEFAULT_CONFIDENCE);
      const sampleMs = Number(req.body?.sample_ms ?? DEFAULT_SAMPLE_MS);
      const confirmCount = Number(req.body?.confirm_count ?? DEFAULT_CONFIRM_COUNT);

      if (!frames) {
        return res.status(503).json({
          ok: false,
          error: 'Frame source not available — shared frame source required',
        });
      }

      // Attach camera to monitor for sampling
      const monitor = await manager.start({
        tableId: table.table_id,
        tableName: table.name,
        cameraId: camera.id,
        cameraName: camera.name,
        store,
        frames,
        detectorId,
        confidence,
        sampleMs,
        confirmCount,
      });

      // Store camera reference for the sampling loop
      monitor._camera = camera;

      res.json({
        ok: true,
        monitor: {
          id: monitor.id,
          tableId: monitor.tableId,
          tableName: monitor.tableName,
          cameraId: monitor.cameraId,
          cameraName: monitor.cameraName,
          state: monitor.state,
          frameCount: monitor.frameCount,
          config: {
            detectorId,
            confidence,
            sampleMs,
            confirmCount,
          },
        },
        message: `Frame state monitoring started for "${table.name}"`,
      });
    } catch (err) {
      res.status(err.status || 500).json({ ok: false, error: err.message });
    }
  });

  // ─── POST /frame-state/:tableId/disable ───────────────────────────

  router.post('/:tableId/disable', (req, res) => {
    const ok = manager.stop(req.params.tableId);
    res.json({ ok: true, stopped: ok, tableId: req.params.tableId });
  });

  // ─── GET /frame-state/:tableId ────────────────────────────────────

  router.get('/:tableId', (req, res) => {
    const monitor = manager.getMonitor(req.params.tableId);
    if (!monitor) {
      return res.status(404).json({ ok: false, error: 'No active monitor for this table' });
    }
    res.json({
      ok: true,
      monitor: {
        id: monitor.id,
        tableId: monitor.tableId,
        tableName: monitor.tableName,
        cameraId: monitor.cameraId,
        cameraName: monitor.cameraName,
        running: monitor.running,
        state: monitor.state,
        lastSample: monitor.lastSample ? {
          totalBalls: monitor.lastBallCount,
          countsByClass: monitor.lastCountsByClass,
          timestamp: monitor.lastSample.timestamp,
          image_size: monitor.lastSample.image_size,
          timings_ms: monitor.lastSample.timings_ms,
        } : null,
        startedAt: monitor.startedAt,
        error: monitor.error,
        samples: monitor.samples,
        frameCount: monitor.frameCount,
      },
    });
  });

  // ─── GET /frame-state ─────────────────────────────────────────────

  router.get('/', (_req, res) => {
    res.json({
      ok: true,
      monitors: manager.listMonitors(),
    });
  });

  // ─── GET /frame-state/events ──────────────────────────────────────

  router.get('/events', (req, res) => {
    const limit = Number(req.query.limit || 100);
    res.json({ ok: true, events: manager.listEvents(limit) });
  });

  // ─── DELETE /frame-state/events ───────────────────────────────────

  router.delete('/events', (_req, res) => {
    manager.clearEvents();
    res.json({ ok: true, events: [] });
  });

  router.stopAll = () => {
    manager.stopAll();
  };

  router.manager = manager;

  return router;
}
