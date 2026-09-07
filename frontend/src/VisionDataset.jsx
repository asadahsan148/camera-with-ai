import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api } from './api';
import DatasetAnnotator from './DatasetAnnotator';

const INTERVALS = [
  { label: '1 FPS', ms: 1000 },
  { label: '2 FPS', ms: 500 },
  { label: '5 FPS', ms: 200 },
];

export default function VisionDataset({ cameras }) {
  const usable = useMemo(() => (cameras || []).filter((c) => c.rtspUrl), [cameras]);
  const [cameraId, setCameraId] = useState('');
  const [tables, setTables] = useState([]);
  const [tableId, setTableId] = useState('');
  const [preview, setPreview] = useState(null);
  const [stats, setStats] = useState(null);
  const [mode, setMode] = useState('capture'); // capture | annotate
  const [busy, setBusy] = useState('');
  const [note, setNote] = useState('');
  const [error, setError] = useState('');
  const [sampling, setSampling] = useState(false);
  const [intervalMs, setIntervalMs] = useState(1000);
  const [sessionCaptured, setSessionCaptured] = useState(0);
  const [sessionSkipped, setSessionSkipped] = useState(0);
  const [captureFps, setCaptureFps] = useState(0);

  const samplingRef = useRef(false);
  const fpsWindow = useRef([]);

  const activeTable = tables.find((t) => t.table_id === tableId) || null;
  const calibrated = activeTable?.calibration_status === 'calibrated';

  useEffect(() => {
    if (usable.length === 0) {
      if (cameraId) setCameraId('');
      return;
    }
    if (!usable.some((c) => c.id === cameraId)) setCameraId(usable[0].id);
  }, [usable, cameraId]);

  const loadTables = useCallback(async (camId) => {
    if (!camId) return;
    const data = await api.listTables(camId);
    setTables(data.tables || []);
    setTableId((prev) => {
      if (data.tables?.some((t) => t.table_id === prev)) return prev;
      const calib = data.tables?.find((t) => t.calibration_status === 'calibrated');
      return calib?.table_id || data.tables?.[0]?.table_id || '';
    });
  }, []);

  useEffect(() => {
    setPreview(null);
    if (cameraId) loadTables(cameraId).catch((e) => setError(e.message));
  }, [cameraId, loadTables]);

  const refreshStats = useCallback(async () => {
    const st = await api.datasetStats();
    setStats(st);
  }, []);

  useEffect(() => {
    refreshStats().catch(() => {});
  }, [refreshStats]);

  const refreshPreview = useCallback(async () => {
    if (!tableId || !calibrated) return;
    try {
      const data = await api.datasetPreview(tableId);
      setPreview(data);
      setError('');
    } catch (err) {
      setError(err.message);
    }
  }, [tableId, calibrated]);

  useEffect(() => {
    refreshPreview();
  }, [refreshPreview]);

  async function captureOnce({ force = false } = {}) {
    if (!tableId) return null;
    setBusy('Capturing…');
    setError('');
    try {
      const data = await api.datasetCapture({ table_id: tableId, force });
      if (data.stats) setStats(data.stats);
      if (data.skipped) {
        setSessionSkipped((n) => n + 1);
        setNote(data.reason || 'Skipped near-duplicate');
      } else {
        setSessionCaptured((n) => n + 1);
        setNote(`Saved ${data.image?.image_id?.slice(0, 8)}…`);
        const now = Date.now();
        fpsWindow.current.push(now);
        fpsWindow.current = fpsWindow.current.filter((t) => now - t < 5000);
        if (fpsWindow.current.length > 1) {
          const span = fpsWindow.current[fpsWindow.current.length - 1] - fpsWindow.current[0];
          setCaptureFps(Number((((fpsWindow.current.length - 1) * 1000) / span).toFixed(2)));
        }
      }
      await refreshPreview();
      return data;
    } catch (err) {
      setError(err.message);
      setSampling(false);
      samplingRef.current = false;
      return null;
    } finally {
      setBusy('');
    }
  }

  useEffect(() => {
    samplingRef.current = sampling;
    if (!sampling) return undefined;
    let stop = false;
    (async function loop() {
      while (!stop && samplingRef.current) {
        const started = Date.now();
        await captureOnce();
        const wait = Math.max(0, intervalMs - (Date.now() - started));
        await new Promise((r) => setTimeout(r, wait));
      }
    })();
    return () => {
      stop = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sampling, intervalMs, tableId]);

  async function exportDataset() {
    setBusy('Exporting…');
    try {
      const data = await api.datasetExport();
      setNote(
        `Export ready: ${data.folder_name}` +
          (data.zip_name ? ` (+ ${data.zip_name})` : '') +
          ` · ${data.images} images`
      );
      await refreshStats();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy('');
    }
  }

  if (mode === 'annotate') {
    return <DatasetAnnotator onBack={() => setMode('capture')} />;
  }

  return (
    <div className="dataset">
      <section className="panel">
        <div className="panel-head">
          <h2>Vision dataset</h2>
          <span>dev / Phase 2</span>
        </div>
        <p className="hint">
          Capture perspective-corrected frames from an existing Phase 1 calibration. Labels are
          manual only — no HSV / legacy detector.
        </p>

        <div className="dataset-controls">
          <label className="field">
            Camera
            <select value={cameraId} onChange={(e) => setCameraId(e.target.value)}>
              <option value="">Select camera…</option>
              {usable.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.name}
                  {c.channel != null ? ` · CH ${c.channel}` : ''}
                </option>
              ))}
            </select>
          </label>

          <label className="field">
            Table
            <select value={tableId} onChange={(e) => setTableId(e.target.value)}>
              <option value="">Select table…</option>
              {tables.map((t) => (
                <option key={t.table_id} value={t.table_id}>
                  {t.name} · {t.calibration_status}
                </option>
              ))}
            </select>
          </label>

          <label className="field">
            Auto interval
            <select
              value={intervalMs}
              onChange={(e) => setIntervalMs(Number(e.target.value))}
              disabled={sampling}
            >
              {INTERVALS.map((o) => (
                <option key={o.ms} value={o.ms}>
                  {o.label}
                </option>
              ))}
            </select>
          </label>
        </div>

        {!calibrated && tableId && (
          <p className="hint warn">Selected table is not calibrated — finish Phase 1 first.</p>
        )}

        <div className="calib-actions">
          <button
            type="button"
            className="btn ghost"
            onClick={refreshPreview}
            disabled={!calibrated || Boolean(busy)}
          >
            Refresh preview
          </button>
          <button
            type="button"
            className="btn primary"
            onClick={() => captureOnce()}
            disabled={!calibrated || sampling || Boolean(busy)}
          >
            Capture Frame
          </button>
          <button
            type="button"
            className="btn ghost"
            onClick={() => captureOnce({ force: true })}
            disabled={!calibrated || sampling || Boolean(busy)}
            title="Save even if similar to an existing frame"
          >
            Force capture
          </button>
          {!sampling ? (
            <button
              type="button"
              className="btn primary"
              onClick={() => {
                setSessionCaptured(0);
                setSessionSkipped(0);
                setSampling(true);
              }}
              disabled={!calibrated || Boolean(busy)}
            >
              Start Capture
            </button>
          ) : (
            <button type="button" className="btn ghost" onClick={() => setSampling(false)}>
              Stop Capture
            </button>
          )}
          <button type="button" className="btn ghost" onClick={() => setMode('annotate')}>
            Open annotation
          </button>
          <button type="button" className="btn ghost" onClick={exportDataset} disabled={Boolean(busy)}>
            Export dataset
          </button>
        </div>

        {(busy || note || error) && (
          <div className={`banner ${error ? 'bad' : 'ok'}`}>{error || busy || note}</div>
        )}
      </section>

      <section className="calib-split">
        <div className="panel calib-view">
          <div className="panel-head">
            <h2>Raw + ROI</h2>
            <span>{preview?.source?.resolution || '—'}</span>
          </div>
          {preview?.raw_image ? (
            <div className="calib-stage">
              <img src={preview.raw_image} alt="Raw with ROI" className="calib-img" />
            </div>
          ) : (
            <div className="empty stage-empty">
              <p>Select a calibrated table to preview.</p>
            </div>
          )}
        </div>
        <div className="panel calib-view">
          <div className="panel-head">
            <h2>Perspective table</h2>
            <span>
              {preview?.output
                ? `${preview.output.output_width}×${preview.output.output_height}`
                : '—'}
            </span>
          </div>
          {preview?.warped_image ? (
            <div className="calib-stage">
              <img src={preview.warped_image} alt="Warped table" className="calib-img" />
            </div>
          ) : (
            <div className="empty stage-empty">
              <p>Warped view appears here.</p>
            </div>
          )}
        </div>
      </section>

      <section className="panel calib-metrics">
        <div className="panel-head">
          <h2>Dataset status</h2>
          <span className={sampling ? 'tag ok' : 'tag'}>{sampling ? 'capturing' : 'idle'}</span>
        </div>
        <dl className="metrics-grid">
          <div>
            <dt>Frames captured (session)</dt>
            <dd>{sessionCaptured}</dd>
          </div>
          <div>
            <dt>Skipped duplicates</dt>
            <dd>{sessionSkipped}</dd>
          </div>
          <div>
            <dt>Capture FPS</dt>
            <dd>{captureFps || '—'}</dd>
          </div>
          <div>
            <dt>Total images</dt>
            <dd>{stats?.total_images ?? 0}</dd>
          </div>
          <div>
            <dt>Annotated</dt>
            <dd>{stats?.annotated_images ?? 0}</dd>
          </div>
          <div>
            <dt>Unannotated</dt>
            <dd>{stats?.unannotated_images ?? 0}</dd>
          </div>
          <div>
            <dt>Total boxes</dt>
            <dd>{stats?.total_annotations ?? 0}</dd>
          </div>
          <div>
            <dt>Dataset size</dt>
            <dd>{stats ? `${stats.dataset_size_mb} MB` : '—'}</dd>
          </div>
          <div>
            <dt>Storage</dt>
            <dd title={stats?.root}>{stats?.root ? 'datasets/scos-v1' : '—'}</dd>
          </div>
        </dl>

        {stats?.annotations_by_class && (
          <div className="class-counts">
            {Object.entries(stats.annotations_by_class).map(([name, n]) => (
              <span key={name} className="tag">
                {name} {n}
              </span>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
