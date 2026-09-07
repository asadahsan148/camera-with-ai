/**
 * Persistent store for table calibration configs.
 *
 * Calibration is written to disk (backend/data/tables.json). Cameras themselves
 * live in an in-memory Map and get NEW UUIDs after every backend restart /
 * "Load all channels". Tables therefore keep a stable camera_key
 * (`nvrIp|chN`) so a saved calibration can be rebound to the current camera
 * UUID without losing corners / matrix / timestamp.
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { randomUUID } from 'crypto';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_FILE = path.join(HERE, '..', 'data', 'tables.json');

export const CORNER_LABELS = ['top-left', 'top-right', 'bottom-right', 'bottom-left'];

export const DEFAULTS = {
  surfaceWidth: 1200,
  aspect: 2.0,
  margin: 0.1,
};

function nowIso() {
  return new Date().toISOString();
}

/** Stable identity that survives camera UUID churn. */
export function stableCameraKey(camera) {
  if (!camera) return null;
  const host = camera.nvrIp || camera.ip;
  if (!host) return null;
  const ch = camera.channel != null && camera.channel !== '' ? String(camera.channel) : 'x';
  return `${host}|ch${ch}`;
}

export function cameraIdentity(camera) {
  if (!camera) return {};
  return {
    camera_id: camera.id,
    camera_name: camera.name || null,
    camera_ip: camera.ip || null,
    camera_nvr_ip: camera.nvrIp || camera.ip || null,
    camera_channel: camera.channel ?? null,
    camera_key: stableCameraKey(camera),
  };
}

export class TableStore {
  constructor(file = DEFAULT_FILE) {
    this.file = file;
    this.tables = new Map();
    this.load();
  }

  load() {
    try {
      const raw = fs.readFileSync(this.file, 'utf8');
      const parsed = JSON.parse(raw);
      for (const t of parsed.tables || []) {
        if (t?.table_id) this.tables.set(t.table_id, t);
      }
    } catch (err) {
      if (err.code !== 'ENOENT') {
        console.warn(`[tables] could not read ${this.file}: ${err.message}`);
      }
    }
  }

  save() {
    const payload = {
      version: 2,
      updated_at: nowIso(),
      tables: [...this.tables.values()],
    };
    fs.mkdirSync(path.dirname(this.file), { recursive: true });
    const tmp = `${this.file}.${process.pid}.tmp`;
    fs.writeFileSync(tmp, JSON.stringify(payload, null, 2), 'utf8');
    fs.renameSync(tmp, this.file);
  }

  list(cameraId) {
    const all = [...this.tables.values()];
    const rows = cameraId ? all.filter((t) => t.camera_id === cameraId) : all;
    return this._sort(rows);
  }

  /**
   * Tables for a live camera, rebound after UUID churn.
   * Matches by camera_id, then camera_key, then legacy camera_name.
   */
  listForCamera(camera, { pruneEmptyDupes = true } = {}) {
    if (!camera?.id) return [];
    const key = stableCameraKey(camera);
    const name = camera.name || null;
    let changed = false;

    const matches = [...this.tables.values()].filter((t) => {
      if (t.camera_id === camera.id) return true;
      if (key && t.camera_key && t.camera_key === key) return true;
      // Legacy rows saved before camera_key existed
      if (!t.camera_key && name && t.camera_name === name) return true;
      return false;
    });

    const rebound = matches.map((t) => {
      const patch = {};
      if (t.camera_id !== camera.id) patch.camera_id = camera.id;
      if (key && t.camera_key !== key) patch.camera_key = key;
      if (camera.ip && t.camera_ip !== camera.ip) patch.camera_ip = camera.ip;
      if ((camera.nvrIp || camera.ip) && t.camera_nvr_ip !== (camera.nvrIp || camera.ip)) {
        patch.camera_nvr_ip = camera.nvrIp || camera.ip;
      }
      if (camera.channel != null && t.camera_channel !== camera.channel) {
        patch.camera_channel = camera.channel;
      }
      if (name && t.camera_name !== name) patch.camera_name = name;
      if (Object.keys(patch).length === 0) return t;
      changed = true;
      return this.update(t.table_id, patch);
    });

    let rows = rebound.filter(Boolean);

    if (pruneEmptyDupes) {
      const removed = this._pruneEmptyDuplicates(rows);
      if (removed) {
        changed = true;
        rows = this.listForCamera(camera, { pruneEmptyDupes: false });
      }
    }

    if (changed) this.save();
    return this._sort(rows);
  }

  /** Drop never-calibrated duplicates that share camera_key + name with a calibrated row. */
  _pruneEmptyDuplicates(rows) {
    const bySlot = new Map();
    for (const t of rows) {
      const slot = `${t.camera_key || t.camera_id}::${t.name || ''}`;
      if (!bySlot.has(slot)) bySlot.set(slot, []);
      bySlot.get(slot).push(t);
    }

    let removed = false;
    for (const group of bySlot.values()) {
      if (group.length < 2) continue;
      const calibrated = group.filter((t) => t.corners_normalized && t.perspective_matrix);
      const empty = group.filter((t) => !t.corners_normalized && !t.perspective_matrix);

      // Prefer calibrated rows; drop empty siblings.
      if (calibrated.length > 0 && empty.length > 0) {
        for (const t of empty) {
          this.tables.delete(t.table_id);
          removed = true;
          console.log(`[tables] pruned empty duplicate ${t.name} (${t.table_id})`);
        }
        continue;
      }

      // No calibrated row: keep the oldest empty, drop the rest (restart churn).
      if (calibrated.length === 0 && empty.length > 1) {
        const sorted = [...empty].sort((a, b) =>
          String(a.created_at || '').localeCompare(String(b.created_at || ''))
        );
        for (const t of sorted.slice(1)) {
          this.tables.delete(t.table_id);
          removed = true;
          console.log(`[tables] pruned duplicate unfinished ${t.name} (${t.table_id})`);
        }
      }
    }
    return removed;
  }

  _sort(rows) {
    return [...rows].sort(
      (a, b) =>
        String(a.camera_key || a.camera_id).localeCompare(String(b.camera_key || b.camera_id)) ||
        String(a.name).localeCompare(String(b.name)) ||
        String(a.created_at || '').localeCompare(String(b.created_at || ''))
    );
  }

  get(tableId) {
    return this.tables.get(tableId) || null;
  }

  /**
   * Point a saved table at the live camera that currently owns its channel.
   * Used by diagnostics when the UUID on disk is stale.
   */
  rebindTable(table, cameras) {
    if (!table) return null;
    const list = Array.isArray(cameras) ? cameras : [];
    const byId = list.find((c) => c.id === table.camera_id);
    if (byId) {
      // Still refresh stable fields if missing
      if (!table.camera_key && stableCameraKey(byId)) {
        return this.update(table.table_id, cameraIdentity(byId));
      }
      return table;
    }

    let match = null;
    if (table.camera_key) {
      match = list.find((c) => stableCameraKey(c) === table.camera_key);
    }
    if (!match && table.camera_name) {
      match = list.find((c) => c.name === table.camera_name);
    }
    if (!match && table.camera_ip != null && table.camera_channel != null) {
      match = list.find(
        (c) =>
          (c.ip === table.camera_ip || c.nvrIp === table.camera_ip || c.ip === table.camera_nvr_ip) &&
          Number(c.channel) === Number(table.camera_channel)
      );
    }
    if (!match) return table;
    return this.update(table.table_id, cameraIdentity(match));
  }

  create({ camera_id, camera_name, name, surface_width, aspect, margin, camera }) {
    const identity = camera
      ? cameraIdentity(camera)
      : {
          camera_id,
          camera_name: camera_name || null,
          camera_ip: null,
          camera_nvr_ip: null,
          camera_channel: null,
          camera_key: null,
        };
    if (!identity.camera_id) throw new Error('camera_id is required');

    const existing = this.listForCamera(
      camera || { id: identity.camera_id, name: identity.camera_name },
      { pruneEmptyDupes: false }
    );
    const table = {
      table_id: randomUUID(),
      ...identity,
      camera_name: camera_name || identity.camera_name,
      name: name || `Table ${existing.length + 1}`,
      corners: null,
      corners_normalized: null,
      reference_width: null,
      reference_height: null,
      perspective_matrix: null,
      surface_width: Number(surface_width) || DEFAULTS.surfaceWidth,
      aspect: Number(aspect) || DEFAULTS.aspect,
      margin: margin === undefined ? DEFAULTS.margin : Number(margin),
      output_width: null,
      output_height: null,
      calibration_timestamp: null,
      created_at: nowIso(),
      updated_at: nowIso(),
    };
    this.tables.set(table.table_id, table);
    this.save();
    return table;
  }

  update(tableId, patch) {
    const table = this.tables.get(tableId);
    if (!table) return null;
    const next = { ...table, ...patch, table_id: table.table_id, updated_at: nowIso() };
    this.tables.set(tableId, next);
    this.save();
    return next;
  }

  /** Store the four clicked points plus the matrix computed for them. */
  saveCalibration(tableId, { corners, referenceWidth, referenceHeight, matrix, output, camera }) {
    const table = this.tables.get(tableId);
    if (!table) return null;

    const normalized = corners.map(([x, y]) => [x / referenceWidth, y / referenceHeight]);
    const identity = camera ? cameraIdentity(camera) : {};
    return this.update(tableId, {
      ...identity,
      corners: corners.map(([x, y]) => [Number(x.toFixed(2)), Number(y.toFixed(2))]),
      corners_normalized: normalized.map(([x, y]) => [Number(x.toFixed(6)), Number(y.toFixed(6))]),
      reference_width: referenceWidth,
      reference_height: referenceHeight,
      perspective_matrix: matrix,
      surface_width: output.surface_width,
      aspect: output.aspect,
      margin: output.margin,
      output_width: output.output_width,
      output_height: output.output_height,
      calibration_timestamp: nowIso(),
    });
  }

  resetCalibration(tableId) {
    const table = this.tables.get(tableId);
    if (!table) return null;
    return this.update(tableId, {
      corners: null,
      corners_normalized: null,
      reference_width: null,
      reference_height: null,
      perspective_matrix: null,
      output_width: null,
      output_height: null,
      calibration_timestamp: null,
    });
  }

  remove(tableId) {
    const ok = this.tables.delete(tableId);
    if (ok) this.save();
    return ok;
  }
}

/** Corners rescaled from the calibration resolution to the current frame. */
export function cornersForFrame(table, frameWidth, frameHeight) {
  if (!table?.corners_normalized || !frameWidth || !frameHeight) return null;
  return table.corners_normalized.map(([x, y]) => [x * frameWidth, y * frameHeight]);
}

export function calibrationStatus(table) {
  if (!table) return 'missing';
  if (!table.corners_normalized || !table.perspective_matrix) return 'not-calibrated';
  return 'calibrated';
}
