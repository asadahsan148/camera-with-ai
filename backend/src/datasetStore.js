/**
 * SCOS Phase 2 — vision dataset filesystem store.
 *
 * Separate from Phase 1 calibration JSON (backend/data/tables.json).
 * Root: <repo>/datasets/scos-v1/
 *
 * Layout:
 *   images/{id}.jpg          perspective-corrected training frames
 *   meta/{id}.json           capture metadata
 *   annotations/{id}.json    manual bounding boxes
 *   classes.json             class definitions (YOLO-ready ids)
 *   index.json               catalog
 *   exports/                 portable packages
 *
 * Annotations use absolute pixel xyxy boxes + class name.
 * Export converts to YOLO txt (normalized cx,cy,w,h) without training.
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { randomUUID, createHash } from 'crypto';
import { spawn } from 'child_process';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(HERE, '..', '..');
export const DEFAULT_DATASET_ROOT = path.join(REPO_ROOT, 'datasets', 'scos-v1');

export const DATASET_CLASSES = [
  { id: 0, name: 'red', color: '#e74c3c' },
  { id: 1, name: 'yellow', color: '#f1c40f' },
  { id: 2, name: 'green', color: '#27ae60' },
  { id: 3, name: 'brown', color: '#8B4513' },
  { id: 4, name: 'blue', color: '#3498db' },
  { id: 5, name: 'pink', color: '#ff69b4' },
  { id: 6, name: 'black', color: '#1a1a1a' },
  { id: 7, name: 'cue_ball', color: '#ecf0f1' },
];

const CLASS_BY_NAME = Object.fromEntries(DATASET_CLASSES.map((c) => [c.name, c]));

function nowIso() {
  return new Date().toISOString();
}

function ensureDir(dir) {
  fs.mkdirSync(dir, { recursive: true });
}

function readJson(file, fallback = null) {
  try {
    return JSON.parse(fs.readFileSync(file, 'utf8'));
  } catch (err) {
    if (err.code === 'ENOENT') return fallback;
    throw err;
  }
}

function writeJsonAtomic(file, data) {
  ensureDir(path.dirname(file));
  const tmp = `${file}.${process.pid}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(data, null, 2), 'utf8');
  fs.renameSync(tmp, file);
}

/** Average-hash on raw JPEG bytes via crude luminance subsample (no sharp/canvas dep). */
export function jpegAverageHash(jpegBuffer, size = 16) {
  // Decode enough of the JPEG via a tiny ffmpeg one-shot is heavy;
  // instead hash file content + size as fallback when pixels unavailable,
  // and prefer perceptual hash computed from a PPM pipe when provided.
  // For capture path we pass precomputed hash from the warp worker path —
  // here: content fingerprint used only as emergency.
  return createHash('sha1').update(jpegBuffer).digest('hex').slice(0, 16);
}

/** Hamming distance between two hex hashes of equal length. */
export function hammingHex(a, b) {
  if (!a || !b || a.length !== b.length) return 64;
  let dist = 0;
  for (let i = 0; i < a.length; i += 1) {
    let x = parseInt(a[i], 16) ^ parseInt(b[i], 16);
    while (x) {
      dist += x & 1;
      x >>= 1;
    }
  }
  return dist;
}

/**
 * Compute a 16x16 average-hash from an image buffer using ffmpeg raw gray.
 * Returns hex string (64 chars for 16x16 = 256 bits → we pack to 64 hex).
 */
export function computePerceptualHash(jpegBuffer) {
  return new Promise((resolve, reject) => {
    const args = [
      '-hide_banner',
      '-loglevel',
      'error',
      '-i',
      'pipe:0',
      '-vf',
      'scale=16:16,format=gray',
      '-f',
      'rawvideo',
      '-pix_fmt',
      'gray',
      'pipe:1',
    ];
    const ff = spawn('ffmpeg', args, { windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'] });
    const chunks = [];
    let err = '';
    ff.stdout.on('data', (c) => chunks.push(c));
    ff.stderr.on('data', (d) => {
      err += d.toString();
    });
    ff.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(err.trim() || `ffmpeg hash failed (${code})`));
        return;
      }
      const pixels = Buffer.concat(chunks);
      if (pixels.length < 256) {
        reject(new Error('hash frame too small'));
        return;
      }
      let sum = 0;
      for (let i = 0; i < 256; i += 1) sum += pixels[i];
      const avg = sum / 256;
      let bits = '';
      for (let i = 0; i < 256; i += 1) bits += pixels[i] >= avg ? '1' : '0';
      let hex = '';
      for (let i = 0; i < 256; i += 4) {
        hex += parseInt(bits.slice(i, i + 4), 2).toString(16);
      }
      resolve(hex);
    });
    ff.stdin.write(jpegBuffer);
    ff.stdin.end();
  });
}

export class DatasetStore {
  constructor(root = DEFAULT_DATASET_ROOT) {
    this.root = root;
    this.imagesDir = path.join(root, 'images');
    this.metaDir = path.join(root, 'meta');
    this.annDir = path.join(root, 'annotations');
    this.exportsDir = path.join(root, 'exports');
    this.indexFile = path.join(root, 'index.json');
    this.classesFile = path.join(root, 'classes.json');
    this._ensureLayout();
  }

  _ensureLayout() {
    for (const d of [this.imagesDir, this.metaDir, this.annDir, this.exportsDir]) ensureDir(d);
    if (!fs.existsSync(this.classesFile)) {
      writeJsonAtomic(this.classesFile, {
        version: 1,
        format: 'scos-bbox-v1',
        yolo_compatible: true,
        classes: DATASET_CLASSES,
        note: 'Manual bounding-box labels. Export writes YOLO txt beside images.',
      });
    }
    if (!fs.existsSync(this.indexFile)) {
      writeJsonAtomic(this.indexFile, {
        version: 1,
        name: 'scos-v1',
        created_at: nowIso(),
        updated_at: nowIso(),
        images: [],
      });
    }
    if (!fs.existsSync(path.join(this.root, 'README.md'))) {
      fs.writeFileSync(
        path.join(this.root, 'README.md'),
        [
          '# SCOS vision dataset (scos-v1)',
          '',
          'Captured from calibrated CCTV perspective views.',
          'Annotations are manual bounding boxes — not HSV/auto labels.',
          '',
          'Do not randomly split consecutive near-duplicate frames across train/val/test.',
          'Prefer session/time-based splits later.',
          '',
        ].join('\n'),
        'utf8'
      );
    }
  }

  _readIndex() {
    return readJson(this.indexFile, { version: 1, name: 'scos-v1', images: [] });
  }

  _writeIndex(index) {
    index.updated_at = nowIso();
    writeJsonAtomic(this.indexFile, index);
  }

  classes() {
    return readJson(this.classesFile, { classes: DATASET_CLASSES });
  }

  list({ annotated, limit = 500, offset = 0 } = {}) {
    const index = this._readIndex();
    let rows = [...(index.images || [])].reverse();
    if (annotated === true) rows = rows.filter((r) => r.annotated);
    if (annotated === false) rows = rows.filter((r) => !r.annotated);
    const total = rows.length;
    rows = rows.slice(Number(offset) || 0, (Number(offset) || 0) + (Number(limit) || 500));
    return { total, images: rows, offset: Number(offset) || 0, limit: Number(limit) || 500 };
  }

  get(imageId) {
    const index = this._readIndex();
    const entry = (index.images || []).find((r) => r.image_id === imageId);
    if (!entry) return null;
    const meta = readJson(path.join(this.metaDir, `${imageId}.json`), {});
    const annotation = readJson(path.join(this.annDir, `${imageId}.json`), {
      image_id: imageId,
      boxes: [],
    });
    return { ...entry, meta, annotation };
  }

  imagePath(imageId) {
    return path.join(this.imagesDir, `${imageId}.jpg`);
  }

  readImage(imageId) {
    const p = this.imagePath(imageId);
    if (!fs.existsSync(p)) return null;
    return fs.readFileSync(p);
  }

  /**
   * @returns {{ saved: object } | { skipped: true, reason: string, nearest?: string }}
   */
  async addCapture({
    jpeg,
    camera,
    table,
    sourceSize,
    outputSize,
    timingsMs,
    force = false,
    similarityThreshold = 8,
  }) {
    this._ensureLayout();
    let phash;
    try {
      phash = await computePerceptualHash(jpeg);
    } catch {
      phash = jpegAverageHash(jpeg);
    }

    const index = this._readIndex();
    if (!force) {
      for (const row of index.images || []) {
        if (!row.phash) continue;
        const dist = hammingHex(phash, row.phash);
        if (dist <= similarityThreshold) {
          return {
            skipped: true,
            reason: `Near-duplicate of ${row.image_id} (hash distance ${dist})`,
            nearest: row.image_id,
            distance: dist,
          };
        }
      }
    }

    const imageId = randomUUID();
    const calibrationId =
      table.calibration_timestamp || table.updated_at || table.table_id || null;

    const meta = {
      image_id: imageId,
      camera_id: camera.id,
      camera_name: camera.name || null,
      camera_key: table.camera_key || null,
      table_id: table.table_id,
      table_name: table.name || null,
      timestamp: nowIso(),
      original_resolution: sourceSize
        ? { width: sourceSize.width, height: sourceSize.height }
        : null,
      perspective_output_resolution: outputSize
        ? { width: outputSize.width, height: outputSize.height }
        : null,
      calibration_id: calibrationId,
      calibration_timestamp: table.calibration_timestamp || null,
      surface_width: table.surface_width,
      aspect: table.aspect,
      margin: table.margin,
      output_width: table.output_width,
      output_height: table.output_height,
      timings_ms: timingsMs || null,
      phash,
      bytes: jpeg.length,
    };

    fs.writeFileSync(this.imagePath(imageId), jpeg);
    writeJsonAtomic(path.join(this.metaDir, `${imageId}.json`), meta);
    writeJsonAtomic(path.join(this.annDir, `${imageId}.json`), {
      image_id: imageId,
      width: outputSize?.width || table.output_width || null,
      height: outputSize?.height || table.output_height || null,
      boxes: [],
      updated_at: null,
      annotated: false,
    });

    const entry = {
      image_id: imageId,
      table_id: table.table_id,
      camera_id: camera.id,
      camera_name: camera.name || null,
      table_name: table.name || null,
      timestamp: meta.timestamp,
      annotated: false,
      box_count: 0,
      bytes: jpeg.length,
      phash,
      width: outputSize?.width || null,
      height: outputSize?.height || null,
    };
    index.images = index.images || [];
    index.images.push(entry);
    this._writeIndex(index);

    return { saved: entry, meta };
  }

  saveAnnotation(imageId, { boxes, width, height }) {
    const index = this._readIndex();
    const entry = (index.images || []).find((r) => r.image_id === imageId);
    if (!entry) return null;

    const cleaned = [];
    for (const b of boxes || []) {
      const name = String(b.class || b.label || '').trim();
      if (!CLASS_BY_NAME[name]) {
        throw Object.assign(new Error(`Unknown class: ${name}`), { status: 400 });
      }
      let x1 = Number(b.x1);
      let y1 = Number(b.y1);
      let x2 = Number(b.x2);
      let y2 = Number(b.y2);
      if (![x1, y1, x2, y2].every(Number.isFinite)) {
        throw Object.assign(new Error('Invalid box coordinates'), { status: 400 });
      }
      if (x2 < x1) [x1, x2] = [x2, x1];
      if (y2 < y1) [y1, y2] = [y2, y1];
      if (x2 - x1 < 2 || y2 - y1 < 2) continue;
      cleaned.push({
        class: name,
        class_id: CLASS_BY_NAME[name].id,
        x1: Math.round(x1 * 10) / 10,
        y1: Math.round(y1 * 10) / 10,
        x2: Math.round(x2 * 10) / 10,
        y2: Math.round(y2 * 10) / 10,
      });
    }

    const ann = {
      image_id: imageId,
      width: width || entry.width || null,
      height: height || entry.height || null,
      boxes: cleaned,
      updated_at: nowIso(),
      annotated: cleaned.length > 0,
      format: 'scos-bbox-v1',
    };
    writeJsonAtomic(path.join(this.annDir, `${imageId}.json`), ann);

    entry.annotated = ann.annotated;
    entry.box_count = cleaned.length;
    entry.annotation_updated_at = ann.updated_at;
    this._writeIndex(index);
    return ann;
  }

  remove(imageId) {
    const index = this._readIndex();
    const before = index.images.length;
    index.images = (index.images || []).filter((r) => r.image_id !== imageId);
    if (index.images.length === before) return false;
    for (const file of [
      this.imagePath(imageId),
      path.join(this.metaDir, `${imageId}.json`),
      path.join(this.annDir, `${imageId}.json`),
    ]) {
      try {
        fs.unlinkSync(file);
      } catch {
        /* ignore */
      }
    }
    this._writeIndex(index);
    return true;
  }

  stats() {
    const index = this._readIndex();
    const images = index.images || [];
    const byClass = Object.fromEntries(DATASET_CLASSES.map((c) => [c.name, 0]));
    let totalBoxes = 0;
    let bytes = 0;
    let annotated = 0;

    for (const row of images) {
      bytes += row.bytes || 0;
      if (row.annotated) annotated += 1;
      const ann = readJson(path.join(this.annDir, `${row.image_id}.json`), null);
      if (!ann?.boxes) continue;
      for (const b of ann.boxes) {
        totalBoxes += 1;
        if (byClass[b.class] != null) byClass[b.class] += 1;
      }
    }

    return {
      dataset: 'scos-v1',
      root: this.root,
      total_images: images.length,
      annotated_images: annotated,
      unannotated_images: images.length - annotated,
      total_annotations: totalBoxes,
      annotations_by_class: byClass,
      dataset_bytes: bytes,
      dataset_size_mb: Number((bytes / (1024 * 1024)).toFixed(2)),
      classes: DATASET_CLASSES,
    };
  }

  /**
   * Build a portable package:
   *   images/  *.jpg
   *   labels/  *.txt   (YOLO)
   *   classes.txt
   *   data.yaml
   *   metadata.jsonl
   *   annotations_json/  original json
   */
  async exportPackage() {
    const stamp = new Date().toISOString().replace(/[:.]/g, '-');
    const outDir = path.join(this.exportsDir, `scos-v1-${stamp}`);
    const imgOut = path.join(outDir, 'images');
    const lblOut = path.join(outDir, 'labels');
    const jsonOut = path.join(outDir, 'annotations_json');
    ensureDir(imgOut);
    ensureDir(lblOut);
    ensureDir(jsonOut);

    const index = this._readIndex();
    const metaLines = [];
    let copied = 0;
    let labeled = 0;

    for (const row of index.images || []) {
      const src = this.imagePath(row.image_id);
      if (!fs.existsSync(src)) continue;
      fs.copyFileSync(src, path.join(imgOut, `${row.image_id}.jpg`));
      copied += 1;

      const ann = readJson(path.join(this.annDir, `${row.image_id}.json`), { boxes: [] });
      const meta = readJson(path.join(this.metaDir, `${row.image_id}.json`), {});
      fs.copyFileSync(
        path.join(this.annDir, `${row.image_id}.json`),
        path.join(jsonOut, `${row.image_id}.json`)
      );

      const w = ann.width || row.width || meta.perspective_output_resolution?.width;
      const h = ann.height || row.height || meta.perspective_output_resolution?.height;
      const lines = [];
      if (w && h && ann.boxes?.length) {
        for (const b of ann.boxes) {
          const cls = CLASS_BY_NAME[b.class];
          if (!cls) continue;
          const bw = b.x2 - b.x1;
          const bh = b.y2 - b.y1;
          const cx = (b.x1 + b.x2) / 2 / w;
          const cy = (b.y1 + b.y2) / 2 / h;
          const nw = bw / w;
          const nh = bh / h;
          lines.push(
            `${cls.id} ${cx.toFixed(6)} ${cy.toFixed(6)} ${nw.toFixed(6)} ${nh.toFixed(6)}`
          );
        }
        labeled += 1;
      }
      fs.writeFileSync(path.join(lblOut, `${row.image_id}.txt`), `${lines.join('\n')}${lines.length ? '\n' : ''}`);

      metaLines.push(
        JSON.stringify({
          image_id: row.image_id,
          file: `images/${row.image_id}.jpg`,
          label: `labels/${row.image_id}.txt`,
          annotated: Boolean(ann.boxes?.length),
          box_count: ann.boxes?.length || 0,
          table_id: meta.table_id,
          camera_id: meta.camera_id,
          timestamp: meta.timestamp,
          calibration_id: meta.calibration_id,
        })
      );
    }

    fs.writeFileSync(
      path.join(outDir, 'classes.txt'),
      `${DATASET_CLASSES.map((c) => c.name).join('\n')}\n`,
      'utf8'
    );
    fs.writeFileSync(
      path.join(outDir, 'data.yaml'),
      [
        '# SCOS YOLO-ready export — splits not assigned yet',
        `# Prefer session/time-based train/val/test later (no consecutive-frame leakage)`,
        `path: .`,
        `train: images`,
        `val: images`,
        `names:`,
        ...DATASET_CLASSES.map((c) => `  ${c.id}: ${c.name}`),
        '',
      ].join('\n'),
      'utf8'
    );
    fs.writeFileSync(path.join(outDir, 'metadata.jsonl'), `${metaLines.join('\n')}\n`, 'utf8');
    writeJsonAtomic(path.join(outDir, 'export_info.json'), {
      dataset: 'scos-v1',
      exported_at: nowIso(),
      images: copied,
      labeled_images: labeled,
      format: 'yolo-txt + scos-bbox-json',
      note: 'train/val/test split intentionally not applied — avoid consecutive-frame leakage',
      classes: DATASET_CLASSES,
    });

    // Best-effort zip on Windows
    const zipPath = `${outDir}.zip`;
    try {
      await zipFolder(outDir, zipPath);
    } catch {
      /* folder still valid */
    }

    return {
      folder: outDir,
      zip: fs.existsSync(zipPath) ? zipPath : null,
      images: copied,
      labeled_images: labeled,
    };
  }
}

function zipFolder(folder, zipPath) {
  return new Promise((resolve, reject) => {
    if (process.platform === 'win32') {
      const ps = spawn(
        'powershell.exe',
        [
          '-NoProfile',
          '-Command',
          `Compress-Archive -Path '${folder}\\*' -DestinationPath '${zipPath}' -Force`,
        ],
        { windowsHide: true }
      );
      ps.on('close', (code) => (code === 0 ? resolve() : reject(new Error(`zip exit ${code}`))));
      return;
    }
    reject(new Error('zip not configured'));
  });
}
