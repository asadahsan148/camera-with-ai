/**
 * Benchmark snapshot store for pretrained model tests.
 * Separate from the Phase 2 training dataset.
 *
 * Root: <repo>/datasets/model-tests/
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { randomUUID } from 'crypto';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(HERE, '..', '..');
export const MODEL_TEST_ROOT = path.join(REPO_ROOT, 'datasets', 'model-tests');

function ensureDir(dir) {
  fs.mkdirSync(dir, { recursive: true });
}

function writeJson(file, data) {
  ensureDir(path.dirname(file));
  const tmp = `${file}.${process.pid}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(data, null, 2), 'utf8');
  fs.renameSync(tmp, file);
}

export class ModelTestSnapshotStore {
  constructor(root = MODEL_TEST_ROOT) {
    this.root = root;
    this.indexFile = path.join(root, 'index.json');
    ensureDir(root);
    if (!fs.existsSync(this.indexFile)) {
      writeJson(this.indexFile, { version: 1, snapshots: [] });
    }
  }

  save({
    cleanJpeg,
    annotatedJpeg,
    predictions,
    rawPredictions = null,
    confidenceThreshold,
    modelInfo,
    camera,
    table,
    timings,
    imageSize,
  }) {
    const id = randomUUID();
    const stamp = new Date().toISOString().replace(/[:.]/g, '-');
    const dir = path.join(this.root, `${stamp}_${id.slice(0, 8)}`);
    ensureDir(dir);

    fs.writeFileSync(path.join(dir, 'frame.jpg'), cleanJpeg);
    if (annotatedJpeg?.length) {
      fs.writeFileSync(path.join(dir, 'frame_detections.jpg'), annotatedJpeg);
    }

    const meta = {
      snapshot_id: id,
      timestamp: new Date().toISOString(),
      confidence_threshold: confidenceThreshold,
      model: modelInfo,
      camera_id: camera?.id || null,
      camera_name: camera?.name || null,
      table_id: table?.table_id || null,
      table_name: table?.name || null,
      calibration_timestamp: table?.calibration_timestamp || null,
      timings_ms: timings || null,
      image_size: imageSize || null,
      detection_count: Array.isArray(predictions) ? predictions.length : 0,
      files: {
        frame: 'frame.jpg',
        annotated: annotatedJpeg?.length ? 'frame_detections.jpg' : null,
        predictions: 'predictions.json',
        raw: rawPredictions ? 'raw_predictions.json' : null,
      },
    };

    writeJson(path.join(dir, 'predictions.json'), {
      meta,
      detections: predictions || [],
    });
    if (rawPredictions) {
      writeJson(path.join(dir, 'raw_predictions.json'), rawPredictions);
    }
    writeJson(path.join(dir, 'meta.json'), meta);

    const index = JSON.parse(fs.readFileSync(this.indexFile, 'utf8'));
    index.snapshots = index.snapshots || [];
    index.snapshots.push({
      snapshot_id: id,
      dir: path.basename(dir),
      timestamp: meta.timestamp,
      model_id: modelInfo?.model_id || modelInfo?.id || null,
      table_id: meta.table_id,
      camera_id: meta.camera_id,
      confidence_threshold: confidenceThreshold,
      detection_count: meta.detection_count,
    });
    writeJson(this.indexFile, index);

    return { ...meta, path: dir };
  }

  list(limit = 50) {
    const index = JSON.parse(fs.readFileSync(this.indexFile, 'utf8'));
    const rows = [...(index.snapshots || [])].reverse().slice(0, limit);
    return { snapshots: rows, root: this.root };
  }
}
