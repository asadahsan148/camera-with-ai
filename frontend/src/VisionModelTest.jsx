import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api } from './api';

const CONF_OPTIONS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
const RATE_OPTIONS = [
  { label: '0.5 FPS', ms: 2000 },
  { label: '1 FPS', ms: 1000 },
  { label: '2 FPS', ms: 500 },
];

export default function VisionModelTest({ cameras }) {
  const usable = useMemo(() => (cameras || []).filter((c) => c.rtspUrl), [cameras]);
  const [cameraId, setCameraId] = useState('');
  const [tables, setTables] = useState([]);
  const [tableId, setTableId] = useState('');
  const [confidence, setConfidence] = useState(0.4);
  const [rateMs, setRateMs] = useState(1000);
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState(null);
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');
  const [note, setNote] = useState('');
  const [inferFps, setInferFps] = useState(0);

  const runningRef = useRef(false);
  const fpsWindow = useRef([]);
  const confidenceRef = useRef(confidence);
  confidenceRef.current = confidence;

  const activeTable = tables.find((t) => t.table_id === tableId) || null;
  const calibrated = activeTable?.calibration_status === 'calibrated';

  useEffect(() => {
    api.modelTestStatus()
      .then(setStatus)
      .catch((e) => setError(e.message));
  }, []);

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
    setResult(null);
    if (cameraId) loadTables(cameraId).catch((e) => setError(e.message));
  }, [cameraId, loadTables]);

  const runOnce = useCallback(async () => {
    if (!tableId) return null;
    try {
      const data = await api.modelTestInfer({
        table_id: tableId,
        confidence: confidenceRef.current,
        detector_id: 'roboflow-snooker',
        draw: true,
      });
      setResult(data);
      setError('');
      const now = Date.now();
      fpsWindow.current.push(now);
      fpsWindow.current = fpsWindow.current.filter((t) => now - t < 5000);
      if (fpsWindow.current.length > 1) {
        const span = fpsWindow.current.at(-1) - fpsWindow.current[0];
        setInferFps(Number((((fpsWindow.current.length - 1) * 1000) / span).toFixed(2)));
      }
      return data;
    } catch (err) {
      setError(err.message);
      setRunning(false);
      runningRef.current = false;
      return null;
    }
  }, [tableId]);

  useEffect(() => {
    runningRef.current = running;
    if (!running) return undefined;
    let stop = false;
    (async function loop() {
      while (!stop && runningRef.current) {
        const started = Date.now();
        await runOnce();
        const wait = Math.max(0, rateMs - (Date.now() - started));
        await new Promise((r) => setTimeout(r, wait));
      }
    })();
    return () => {
      stop = true;
    };
  }, [running, rateMs, runOnce]);

  // Re-filter display when confidence changes mid-run without waiting for next cloud call:
  // server already returns detections_all — re-filter client-side for snappy UI.
  const shown = useMemo(() => {
    if (!result) return null;
    const all = result.detections_all || result.detections || [];
    const filtered = all.filter((d) => d.confidence >= confidence);
    const low = all.filter((d) => d.confidence < 0.5);
    const counts = {};
    for (const d of filtered) counts[d.class] = (counts[d.class] || 0) + 1;
    return {
      detections: filtered,
      low_confidence_count: low.length,
      counts_by_class: counts,
      total: filtered.length,
    };
  }, [result, confidence]);

  async function saveSnapshot() {
    if (!tableId) return;
    setBusy('Saving snapshot…');
    try {
      const data = await api.modelTestSnapshot({
        table_id: tableId,
        confidence,
        detector_id: 'roboflow-snooker',
      });
      setNote(`Snapshot saved: ${data.snapshot?.snapshot_id?.slice(0, 8)}… → datasets/model-tests/`);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy('');
    }
  }

  const keyOk = Boolean(status?.api_key_configured);

  return (
    <div className="dataset">
      <section className="panel">
        <div className="panel-head">
          <h2>Pretrained model test</h2>
          <span className={keyOk ? 'tag ok' : 'tag'}>
            {keyOk ? 'API key OK' : 'API key missing'}
          </span>
        </div>
        <p className="hint">
          Benchmarks Roboflow <code>{status?.detector?.model_id || 'snooker-ball-detection-rnhxo-95km5/2'}</code>{' '}
          on the Phase 1 warped table. Diagnostic only — no scoring / tracking.
        </p>
        {!keyOk && (
          <p className="hint warn">
            Put your key in <strong>backend/.env</strong> as <code>ROBOFLOW_API_KEY=...</code> then restart
            the backend. The key never goes to the browser.
          </p>
        )}

        <div className="dataset-controls">
          <label className="field">
            Camera
            <select value={cameraId} onChange={(e) => setCameraId(e.target.value)}>
              <option value="">Select…</option>
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
              <option value="">Select…</option>
              {tables.map((t) => (
                <option key={t.table_id} value={t.table_id}>
                  {t.name} · {t.calibration_status}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            Confidence ≥
            <select value={confidence} onChange={(e) => setConfidence(Number(e.target.value))}>
              {CONF_OPTIONS.map((c) => (
                <option key={c} value={c}>
                  {c.toFixed(2)}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            Infer rate
            <select
              value={rateMs}
              onChange={(e) => setRateMs(Number(e.target.value))}
              disabled={running}
            >
              {RATE_OPTIONS.map((o) => (
                <option key={o.ms} value={o.ms}>
                  {o.label}
                </option>
              ))}
            </select>
          </label>
        </div>

        {!calibrated && tableId && (
          <p className="hint warn">Table not calibrated — finish Phase 1 first.</p>
        )}

        <div className="calib-actions">
          {!running ? (
            <button
              type="button"
              className="btn primary"
              disabled={!calibrated || !keyOk}
              onClick={() => setRunning(true)}
            >
              Start detection
            </button>
          ) : (
            <button type="button" className="btn ghost" onClick={() => setRunning(false)}>
              Stop
            </button>
          )}
          <button
            type="button"
            className="btn ghost"
            disabled={!calibrated || !keyOk || running || Boolean(busy)}
            onClick={() => runOnce()}
          >
            Single frame
          </button>
          <button
            type="button"
            className="btn ghost"
            disabled={!calibrated || !keyOk || Boolean(busy)}
            onClick={saveSnapshot}
          >
            Save Test Snapshot
          </button>
        </div>

        {(busy || note || error) && (
          <div className={`banner ${error ? 'bad' : 'ok'}`}>{error || busy || note}</div>
        )}
      </section>

      <section className="calib-split">
        <div className="panel calib-view">
          <div className="panel-head">
            <h2>Clean warped table</h2>
            <span>
              {result?.source
                ? `${result.source.warped_width}×${result.source.warped_height}`
                : '—'}
            </span>
          </div>
          {result?.clean_image ? (
            <div className="calib-stage">
              <img src={result.clean_image} alt="Clean table" className="calib-img" />
            </div>
          ) : (
            <div className="empty stage-empty">
              <p>Start detection to see the perspective table.</p>
            </div>
          )}
        </div>
        <div className="panel calib-view">
          <div className="panel-head">
            <h2>Detections</h2>
            <span>{shown ? `${shown.total} boxes` : '—'}</span>
          </div>
          {result?.clean_image ? (
            <div className="calib-stage model-det-stage">
              <img src={result.clean_image} alt="Detections" className="calib-img" />
              {shown && result.source && (
                <svg
                  className="calib-overlay"
                  viewBox={`0 0 ${result.source.warped_width} ${result.source.warped_height}`}
                  preserveAspectRatio="xMidYMid meet"
                >
                  {shown.detections.map((d, i) => (
                    <g key={i}>
                      <rect
                        x={d.x}
                        y={d.y}
                        width={d.width}
                        height={d.height}
                        fill="none"
                        stroke={d.confidence < 0.5 ? '#e8b84a' : '#3dd68c'}
                        strokeWidth="2"
                        vectorEffect="non-scaling-stroke"
                      />
                      <text
                        x={d.x + 2}
                        y={Math.max(14, d.y - 4)}
                        fill={d.confidence < 0.5 ? '#e8b84a' : '#3dd68c'}
                        fontSize="16"
                        fontWeight="700"
                      >
                        {d.class} {d.confidence.toFixed(2)}
                      </text>
                    </g>
                  ))}
                </svg>
              )}
            </div>
          ) : (
            <div className="empty stage-empty">
              <p>Boxes appear here after inference.</p>
            </div>
          )}
        </div>
      </section>

      <section className="panel calib-metrics">
        <div className="panel-head">
          <h2>Live stats</h2>
          <span className={running ? 'tag ok' : 'tag'}>{running ? 'running' : 'idle'}</span>
        </div>
        <dl className="metrics-grid">
          <div>
            <dt>Model</dt>
            <dd>{result?.model?.model_id || status?.detector?.model_id || '—'}</dd>
          </div>
          <div>
            <dt>Inference FPS</dt>
            <dd>{inferFps || '—'}</dd>
          </div>
          <div>
            <dt>Inference time</dt>
            <dd>
              {result?.timings_ms?.inference != null ? `${result.timings_ms.inference} ms` : '—'}
            </dd>
          </div>
          <div>
            <dt>Round trip</dt>
            <dd>
              {result?.timings_ms?.request != null ? `${result.timings_ms.request} ms` : '—'}
            </dd>
          </div>
          <div>
            <dt>Total detections</dt>
            <dd>{shown?.total ?? '—'}</dd>
          </div>
          <div>
            <dt>Low conf (&lt;0.50)</dt>
            <dd>{shown?.low_confidence_count ?? '—'}</dd>
          </div>
          <div>
            <dt>Threshold</dt>
            <dd>{confidence.toFixed(2)}</dd>
          </div>
          <div>
            <dt>Raw model classes seen</dt>
            <dd title={(result?.stats?.model_classes_seen || []).join(', ')}>
              {(result?.stats?.model_classes_seen || []).slice(0, 6).join(', ') || '—'}
            </dd>
          </div>
        </dl>
        {shown?.counts_by_class && (
          <div className="class-counts">
            {Object.entries(shown.counts_by_class).map(([name, n]) => (
              <span key={name} className="tag">
                {name} {n}
              </span>
            ))}
          </div>
        )}
        {shown?.detections?.length > 0 && (
          <ul className="det-list">
            {shown.detections
              .slice()
              .sort((a, b) => b.confidence - a.confidence)
              .slice(0, 24)
              .map((d, i) => (
                <li key={i}>
                  <strong>{d.class}</strong> {d.confidence.toFixed(2)}
                  {d.raw_class && d.raw_class !== d.class ? ` · raw:${d.raw_class}` : ''}
                </li>
              ))}
          </ul>
        )}
      </section>
    </div>
  );
}
