/**
 * SCOS Frame State Manager — detects snooker frame start/end.
 *
 * Periodically samples a calibrated table view, runs ball detection,
 * and uses a state machine with hysteresis to detect:
 *   - Frame start (balls racked in starting position)
 *   - Frame end (table cleared)
 *
 * States: IDLE → FRAME_START → IN_FRAME → FRAME_END → IDLE
 *
 * Reuses existing detector adapters, frame source, and table store.
 */

import { EventEmitter } from 'events';
import { randomUUID } from 'crypto';
import { FrameSource } from './frameSource.js';
import { TableStore, cornersForFrame, calibrationStatus } from './tableStore.js';
import { getDetector } from './detectors/index.js';
import { filterByConfidence, countByClass } from './detectors/DetectorAdapter.js';

const AI_URL = process.env.AI_WORKER_URL || 'http://127.0.0.1:5051';
const DEFAULT_SAMPLE_MS = Number(process.env.FRAME_STATE_SAMPLE_MS || 5000);
const DEFAULT_CONFIDENCE = Number(process.env.FRAME_STATE_CONFIDENCE || 0.4);
const DEFAULT_CONFIRM_COUNT = Number(process.env.FRAME_STATE_CONFIRM_COUNT || 3);
const DEFAULT_DETECTOR_ID = process.env.FRAME_STATE_DETECTOR || 'roboflow-snooker';

// Ball count thresholds
const FRAME_START_MIN_REDS = 10;
const FRAME_START_MIN_COLORS = 4;
const FRAME_START_MIN_TOTAL = 18;
const FRAME_END_MAX_BALLS = 1;
// Fallback for generic ball detectors (COCO "sports ball" — no color info)
const FRAME_START_GENERIC_MIN = 15;
const FRAME_END_GENERIC_MAX = 1;

export const FRAME_STATES = {
  IDLE: 'IDLE',
  FRAME_START: 'FRAME_START',
  IN_FRAME: 'IN_FRAME',
  FRAME_END: 'FRAME_END',
};

export class FrameStateManager extends EventEmitter {
  constructor() {
    super();
    this.monitors = new Map();
    this.events = [];
    this.maxEvents = 500;
  }

  listMonitors() {
    return [...this.monitors.values()].map((m) => ({
      id: m.id,
      tableId: m.tableId,
      tableName: m.tableName,
      cameraId: m.cameraId,
      cameraName: m.cameraName,
      running: m.running,
      state: m.state,
      lastSample: m.lastSample,
      startedAt: m.startedAt,
      error: m.error,
      samples: m.samples,
      frameCount: m.frameCount,
    }));
  }

  listEvents(limit = 100) {
    return this.events.slice(0, limit);
  }

  clearEvents() {
    this.events = [];
  }

  getMonitor(tableId) {
    return this.monitors.get(tableId) || null;
  }

  pushEvent(event) {
    const row = {
      id: randomUUID(),
      ts: new Date().toISOString(),
      ...event,
    };
    this.events.unshift(row);
    if (this.events.length > this.maxEvents) this.events.length = this.maxEvents;
    this.emit('event', row);
    console.log(
      `[FrameState] ${row.ts} ${row.type} table=${row.tableName || row.tableId} ${row.message || ''}`
    );
    return row;
  }

  async start({
    tableId,
    tableName,
    cameraId,
    cameraName,
    store,
    frames,
    detectorId = DEFAULT_DETECTOR_ID,
    confidence = DEFAULT_CONFIDENCE,
    sampleMs = DEFAULT_SAMPLE_MS,
    confirmCount = DEFAULT_CONFIRM_COUNT,
  }) {
    if (this.monitors.has(tableId)) {
      return this.monitors.get(tableId);
    }

    const id = randomUUID();
    const monitor = {
      id,
      tableId,
      tableName: tableName || tableId,
      cameraId,
      cameraName: cameraName || cameraId,
      running: true,
      state: FRAME_STATES.IDLE,
      lastSample: null,
      startedAt: new Date().toISOString(),
      error: null,
      samples: 0,
      frameCount: 0,
      timer: null,
      busy: false,
      // Config
      store,
      frames,
      detectorId,
      confidence,
      confirmCount,
      // Hysteresis counters
      startConfirmations: 0,
      endConfirmations: 0,
      // Last detection result
      lastBallCount: 0,
      lastCountsByClass: {},
    };

    this.monitors.set(tableId, monitor);

    this.pushEvent({
      type: 'FRAME_STATE_MONITOR_STARTED',
      tableId,
      tableName: monitor.tableName,
      cameraId,
      cameraName: monitor.cameraName,
      monitorId: id,
      message: 'Frame state monitor started',
    });

    const tick = async () => {
      if (!monitor.running || monitor.busy) return;
      monitor.busy = true;

      try {
        const result = await this._sample(monitor);
        monitor.lastSample = result;
        monitor.samples += 1;
        monitor.error = null;

        this._updateStateMachine(monitor, result);
      } catch (err) {
        monitor.error = err.message;
        if (monitor.samples === 0 || monitor.samples % 10 === 0) {
          this.pushEvent({
            type: 'FRAME_STATE_ERROR',
            tableId,
            tableName: monitor.tableName,
            cameraId,
            cameraName: monitor.cameraName,
            monitorId: id,
            message: err.message,
          });
        }
      } finally {
        monitor.busy = false;
      }
    };

    // First tick after 1s, then interval
    setTimeout(tick, 1000);
    monitor.timer = setInterval(tick, sampleMs);

    return monitor;
  }

  stop(tableId) {
    const monitor = this.monitors.get(tableId);
    if (!monitor) return false;

    monitor.running = false;
    if (monitor.timer) clearInterval(monitor.timer);
    this.monitors.delete(tableId);

    this.pushEvent({
      type: 'FRAME_STATE_MONITOR_STOPPED',
      tableId,
      tableName: monitor.tableName,
      cameraId: monitor.cameraId,
      cameraName: monitor.cameraName,
      monitorId: monitor.id,
      message: 'Frame state monitor stopped',
    });

    return true;
  }

  stopAll() {
    for (const tableId of [...this.monitors.keys()]) {
      this.stop(tableId);
    }
  }

  async _sample(monitor) {
    const { store, frames, cameraId, tableId, detectorId, confidence } = monitor;

    const table = store.get(tableId);
    if (!table) throw new Error(`Table not found: ${tableId}`);

    const camera = store.rebindTable(table, []) ? table._camera : null;
    // We need camera from the external getCamera — stored on monitor
    const cam = monitor._camera || {};
    if (!cam.rtspUrl) throw new Error('Camera not connected');

    const frame = await frames.grab(cam.id, cam.rtspUrl);
    const corners = cornersForFrame(table, frame.width, frame.height);
    if (!corners) throw new Error('Could not scale calibration corners to frame');

    const warp = await this._workerWarp(frame.jpeg, {
      corners,
      surfaceWidth: table.surface_width,
      aspect: table.aspect,
      margin: table.margin,
      label: table.name,
    });

    const warpedJpeg = Buffer.from(warp.warped_jpeg_base64, 'base64');
    const outputSize = {
      width: warp.output?.output_width || table.output_width,
      height: warp.output?.output_height || table.output_height,
    };

    const detector = getDetector(detectorId);
    const result = await detector.detect(warpedJpeg, {
      confidence: 0.01,
      sourceImageSize: outputSize,
    });

    const filtered = filterByConfidence(result.detections || [], confidence);
    const counts = countByClass(filtered);
    const totalBalls = filtered.length;

    return {
      totalBalls,
      countsByClass: counts,
      detections: filtered,
      image_size: outputSize,
      timings_ms: result.timings_ms,
      timestamp: new Date().toISOString(),
    };
  }

  async _workerWarp(jpeg, opts) {
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
      throw new Error(detail || `AI worker HTTP ${res.status}`);
    }
    return data;
  }

  _updateStateMachine(monitor, sample) {
    const { totalBalls, countsByClass } = sample;
    const prevState = monitor.state;

    monitor.lastBallCount = totalBalls;
    monitor.lastCountsByClass = countsByClass;

    const redCount = countsByClass.red || 0;
    const colorCount = (countsByClass.yellow || 0) + (countsByClass.green || 0) +
      (countsByClass.brown || 0) + (countsByClass.blue || 0) +
      (countsByClass.pink || 0) + (countsByClass.black || 0);
    const hasCueBall = (countsByClass.cue_ball || 0) > 0;
    const genericBalls = countsByClass.ball || 0;

    // Use color-specific detection if available, otherwise fall back to generic ball count
    const hasColorInfo = redCount > 0 || colorCount > 0;
    const effectiveTotal = hasColorInfo ? totalBalls : Math.max(totalBalls, genericBalls);

    const isFrameStart = hasColorInfo
      ? (redCount >= FRAME_START_MIN_REDS && colorCount >= FRAME_START_MIN_COLORS &&
         totalBalls >= FRAME_START_MIN_TOTAL)
      : (effectiveTotal >= FRAME_START_GENERIC_MIN);

    const isFrameEnd = hasColorInfo
      ? (totalBalls <= FRAME_END_MAX_BALLS)
      : (effectiveTotal <= FRAME_END_GENERIC_MAX);

    switch (monitor.state) {
      case FRAME_STATES.IDLE:
        if (isFrameStart) {
          monitor.startConfirmations += 1;
          monitor.endConfirmations = 0;
          if (monitor.startConfirmations >= monitor.confirmCount) {
            monitor.state = FRAME_STATES.FRAME_START;
            monitor.frameCount += 1;
            monitor.startConfirmations = 0;
            this.pushEvent({
              type: 'FRAME_START',
              tableId: monitor.tableId,
              tableName: monitor.tableName,
              cameraId: monitor.cameraId,
              cameraName: monitor.cameraName,
              frameNumber: monitor.frameCount,
              ballCount: effectiveTotal,
              countsByClass,
              message: `Frame ${monitor.frameCount} started — ${effectiveTotal} balls detected`,
            });
          }
        } else {
          monitor.startConfirmations = 0;
        }
        break;

      case FRAME_STATES.FRAME_START:
        // Immediately transition to IN_FRAME after start is confirmed
        monitor.state = FRAME_STATES.IN_FRAME;
        break;

      case FRAME_STATES.IN_FRAME:
        if (isFrameEnd) {
          monitor.endConfirmations += 1;
          monitor.startConfirmations = 0;
          if (monitor.endConfirmations >= monitor.confirmCount) {
            monitor.state = FRAME_STATES.FRAME_END;
            this.pushEvent({
              type: 'FRAME_END',
              tableId: monitor.tableId,
              tableName: monitor.tableName,
              cameraId: monitor.cameraId,
              cameraName: monitor.cameraName,
              frameNumber: monitor.frameCount,
              ballCount: effectiveTotal,
              message: `Frame ${monitor.frameCount} ended — table cleared`,
            });
          }
        } else {
          monitor.endConfirmations = 0;
          // Check for new frame start (re-rack)
          if (isFrameStart) {
            monitor.frameCount += 1;
            monitor.state = FRAME_STATES.FRAME_START;
            this.pushEvent({
              type: 'FRAME_START',
              tableId: monitor.tableId,
              tableName: monitor.tableName,
              cameraId: monitor.cameraId,
              cameraName: monitor.cameraName,
              frameNumber: monitor.frameCount,
              ballCount: effectiveTotal,
              countsByClass,
              message: `Frame ${monitor.frameCount} started (re-rack) — ${effectiveTotal} balls detected`,
            });
          }
        }
        break;

      case FRAME_STATES.FRAME_END:
        // Transition back to IDLE
        monitor.state = FRAME_STATES.IDLE;
        monitor.endConfirmations = 0;
        // Check if a new frame is already starting
        if (isFrameStart) {
          monitor.startConfirmations = 1;
        }
        break;
    }

    if (monitor.state !== prevState) {
      this.emit('stateChange', {
        tableId: monitor.tableId,
        tableName: monitor.tableName,
        prevState,
        newState: monitor.state,
        timestamp: new Date().toISOString(),
      });
    }
  }
}
