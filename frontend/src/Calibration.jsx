import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api } from './api';

const CORNER_LABELS = ['top-left', 'top-right', 'bottom-right', 'bottom-left'];
const CORNER_SHORT = ['TL', 'TR', 'BR', 'BL'];

function fmt(value, suffix = '') {
  if (value === null || value === undefined || value === '') return '—';
  return `${value}${suffix}`;
}

export default function Calibration({ cameras }) {
  const usable = useMemo(
    () => (cameras || []).filter((c) => c.rtspUrl),
    [cameras]
  );

  const [cameraId, setCameraId] = useState('');
  const [tables, setTables] = useState([]);
  const [activeId, setActiveId] = useState('');
  const [snapshot, setSnapshot] = useState(null);
  const [draft, setDraft] = useState([]);
  const [preview, setPreview] = useState(null);
  const [diag, setDiag] = useState(null);
  const [live, setLive] = useState(false);
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');
  const [note, setNote] = useState('');
  const [showLegacy, setShowLegacy] = useState(false);

  const svgRef = useRef(null);
  const dragRef = useRef(null);
  const liveRef = useRef(false);

  const active = tables.find((t) => t.table_id === activeId) || null;
  const calibrated = active?.calibration_status === 'calibrated';
  const editing = draft.length > 0 || !calibrated;

  const loadTables = useCallback(async (camId, keepActive = true) => {
    if (!camId) return;
    try {
      const data = await api.listTables(camId);
      setTables(data.tables || []);
      setActiveId((prev) => {
        if (keepActive && data.tables?.some((t) => t.table_id === prev)) return prev;
        return data.tables?.[0]?.table_id || '';
      });
    } catch (err) {
      setError(err.message);
    }
  }, []);

  useEffect(() => {
    if (usable.length === 0) {
      if (cameraId) setCameraId('');
      return;
    }
    // After backend restart / Load all channels, camera UUIDs change —
    // keep the dropdown on a live camera (prefer same channel/name).
    if (!usable.some((c) => c.id === cameraId)) {
      const prev = (cameras || []).find((c) => c.id === cameraId);
      const match =
        usable.find(
          (c) =>
            prev &&
            ((prev.channel != null && c.channel === prev.channel && c.ip === prev.ip) ||
              (prev.name && c.name === prev.name))
        ) || usable[0];
      setCameraId(match.id);
    }
  }, [usable, cameraId, cameras]);

  useEffect(() => {
    setSnapshot(null);
    setDiag(null);
    setPreview(null);
    setDraft([]);
    setLive(false);
    if (cameraId) loadTables(cameraId, false);
  }, [cameraId, loadTables]);

  // When the live camera list is refreshed (new UUIDs), re-pull tables so
  // the server can rebind saved calibrations via camera_key / name.
  useEffect(() => {
    if (cameraId && usable.some((c) => c.id === cameraId)) {
      loadTables(cameraId, true);
    }
  }, [usable, loadTables]); // eslint-disable-line react-hooks/exhaustive-deps -- cameraId handled above

  async function takeSnapshot() {
    if (!cameraId) return;
    setBusy('Grabbing frame from camera…');
    setError('');
    try {
      const data = await api.snapshot(cameraId);
      setSnapshot(data);
      setNote(`Frame ${data.resolution || ''} in ${data.grab_ms} ms`);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy('');
    }
  }

  async function addTable() {
    if (!cameraId) return;
    const camera = usable.find((c) => c.id === cameraId);
    setBusy('Creating table…');
    setError('');
    try {
      const data = await api.createTable({
        camera_id: cameraId,
        camera_name: camera?.name,
        name: `Table ${tables.length + 1}`,
      });
      await loadTables(cameraId);
      setActiveId(data.table.table_id);
      setDraft([]);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy('');
    }
  }

  async function removeTable(tableId) {
    setBusy('Deleting…');
    try {
      await api.deleteTable(tableId);
      await loadTables(cameraId, false);
      setDiag(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy('');
    }
  }

  async function renameTable(tableId, name) {
    try {
      await api.updateTable(tableId, { name });
      await loadTables(cameraId);
    } catch (err) {
      setError(err.message);
    }
  }

  async function saveCalibration() {
    if (!active || draft.length !== 4 || !snapshot) return;
    setBusy('Saving calibration…');
    setError('');
    try {
      const data = await api.calibrateTable(active.table_id, {
        corners: draft.map((p) => [p.x, p.y]),
        reference_width: snapshot.width,
        reference_height: snapshot.height,
      });
      await loadTables(cameraId);
      setDraft([]);
      setPreview(null);
      setNote(
        data.warnings?.length
          ? `Saved with warnings: ${data.warnings.join(' ')}`
          : `Calibration saved for ${data.table.name}`
      );
      await refreshDiagnostics(active.table_id);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy('');
    }
  }

  async function resetCalibration() {
    if (!active) return;
    setBusy('Resetting…');
    try {
      await api.resetTable(active.table_id);
      await loadTables(cameraId);
      setDraft([]);
      setPreview(null);
      setDiag(null);
      setNote('Calibration cleared — click the four corners again.');
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy('');
    }
  }

  const refreshDiagnostics = useCallback(async (tableId) => {
    if (!tableId) return;
    try {
      const data = await api.diagnostics(tableId);
      setDiag(data);
      setError('');
    } catch (err) {
      setError(err.message);
      setLive(false);
    }
  }, []);

  useEffect(() => {
    liveRef.current = live;
    if (!live || !activeId) return undefined;
    let stop = false;
    (async function loop() {
      while (!stop && liveRef.current) {
        await refreshDiagnostics(activeId);
        await new Promise((r) => setTimeout(r, 400));
      }
    })();
    return () => {
      stop = true;
    };
  }, [live, activeId, refreshDiagnostics]);

  // Preview the warp while corners are being placed
  useEffect(() => {
    if (!active || draft.length !== 4 || !snapshot) {
      setPreview(null);
      return undefined;
    }
    let cancelled = false;
    const t = setTimeout(async () => {
      try {
        const data = await api.previewTable(active.table_id, {
          corners: draft.map((p) => [p.x, p.y]),
          reference_width: snapshot.width,
          reference_height: snapshot.height,
        });
        if (!cancelled) {
          setPreview(data);
          setError('');
        }
      } catch (err) {
        if (!cancelled) setError(err.message);
      }
    }, 150);
    return () => {
      cancelled = true;
      clearTimeout(t);
    };
  }, [draft, active, snapshot]);

  function svgPoint(evt) {
    const svg = svgRef.current;
    if (!svg) return null;
    const ctm = svg.getScreenCTM();
    if (!ctm) return null;
    const pt = svg.createSVGPoint();
    pt.x = evt.clientX;
    pt.y = evt.clientY;
    const local = pt.matrixTransform(ctm.inverse());
    return { x: Number(local.x.toFixed(1)), y: Number(local.y.toFixed(1)) };
  }

  function handleStageClick(evt) {
    if (dragRef.current !== null) return;
    if (draft.length >= 4) return;
    const p = svgPoint(evt);
    if (!p) return;
    setDraft((prev) => (prev.length >= 4 ? prev : [...prev, p]));
  }

  function handlePointerDown(index, evt) {
    evt.stopPropagation();
    dragRef.current = index;
    evt.currentTarget.setPointerCapture?.(evt.pointerId);
  }

  function handlePointerMove(evt) {
    if (dragRef.current === null) return;
    const p = svgPoint(evt);
    if (!p) return;
    setDraft((prev) => prev.map((pt, i) => (i === dragRef.current ? p : pt)));
  }

  function handlePointerUp() {
    dragRef.current = null;
  }

  function startEditing() {
    if (!active?.corners) return;
    setDraft(active.corners.map(([x, y]) => ({ x, y })));
  }

  useEffect(() => {
    setDraft([]);
    setPreview(null);
    if (activeId) refreshDiagnostics(activeId);
  }, [activeId, refreshDiagnostics]);

  const rawImage = diag?.raw_image || snapshot?.image || null;
  const frameW = diag?.source?.width || snapshot?.width || 0;
  const frameH = diag?.source?.height || snapshot?.height || 0;

  // Saved corners scaled to the frame currently being shown
  const savedCorners = useMemo(() => {
    if (!active?.corners_normalized || !frameW || !frameH) return null;
    return active.corners_normalized.map(([x, y]) => ({ x: x * frameW, y: y * frameH }));
  }, [active, frameW, frameH]);

  const shownCorners = draft.length > 0 ? draft : savedCorners || [];
  const rightImage = preview?.warped_image || diag?.warped_image || null;
  const output = preview?.output || diag?.output || null;

  return (
    <div className="calib">
      <section className="panel calib-controls">
        <div className="panel-head">
          <h2>Table calibration</h2>
          <span>{tables.length} table(s)</span>
        </div>

        <label className="field">
          Camera
          <select value={cameraId} onChange={(e) => setCameraId(e.target.value)}>
            <option value="">Select a connected camera…</option>
            {usable.map((c) => (
              <option key={c.id} value={c.id}>
                {c.name}
                {c.channel != null ? ` · CH ${c.channel}` : ''}
              </option>
            ))}
          </select>
        </label>

        {usable.length === 0 && (
          <p className="hint">
            No camera with a verified RTSP URL yet. Go to <strong>Cameras</strong>, load channels /
            connect, then come back.
          </p>
        )}

        <div className="calib-tables">
          {tables.map((t) => (
            <div key={t.table_id} className={`calib-table ${t.table_id === activeId ? 'on' : ''}`}>
              <button type="button" className="calib-table-pick" onClick={() => setActiveId(t.table_id)}>
                <strong>{t.name}</strong>
                <small>
                  {t.calibration_status === 'calibrated'
                    ? `calibrated · ${new Date(t.calibration_timestamp).toLocaleString()}`
                    : 'not calibrated'}
                </small>
              </button>
              <div className="calib-table-actions">
                <button
                  type="button"
                  className="btn ghost"
                  onClick={() => {
                    const name = window.prompt('Table name', t.name);
                    if (name) renameTable(t.table_id, name);
                  }}
                >
                  Rename
                </button>
                <button type="button" className="btn ghost" onClick={() => removeTable(t.table_id)}>
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>

        <div className="calib-actions">
          <button type="button" className="btn primary" onClick={addTable} disabled={!cameraId || Boolean(busy)}>
            Add table
          </button>
          <button type="button" className="btn ghost" onClick={takeSnapshot} disabled={!cameraId || Boolean(busy)}>
            Grab frame
          </button>
          <button
            type="button"
            className={`btn ${live ? 'primary' : 'ghost'}`}
            onClick={() => setLive((v) => !v)}
            disabled={!activeId}
          >
            {live ? 'Stop live diagnostics' : 'Start live diagnostics'}
          </button>
        </div>

        {active && (
          <div className="calib-steps">
            <h3>
              {active.name} — click {CORNER_LABELS[draft.length] || 'done'}
            </h3>
            <ol>
              {CORNER_LABELS.map((label, i) => (
                <li key={label} className={draft[i] ? 'done' : ''}>
                  <span>{CORNER_SHORT[i]}</span> {label}
                  {draft[i] ? ` · ${Math.round(draft[i].x)}, ${Math.round(draft[i].y)}` : ''}
                </li>
              ))}
            </ol>
            <p className="hint">
              Click the four corners of the <strong>playing surface</strong> (inner cushion edge).
              Cushions and pockets stay visible via the {Math.round((active.margin ?? 0.1) * 100)}%
              border kept around the surface. Drag a point to fine-tune.
            </p>
            <div className="calib-actions">
              <button
                type="button"
                className="btn ghost"
                onClick={() => setDraft((p) => p.slice(0, -1))}
                disabled={draft.length === 0}
              >
                Undo point
              </button>
              <button type="button" className="btn ghost" onClick={() => setDraft([])} disabled={draft.length === 0}>
                Clear points
              </button>
              {calibrated && draft.length === 0 && (
                <button type="button" className="btn ghost" onClick={startEditing}>
                  Edit saved corners
                </button>
              )}
              <button
                type="button"
                className="btn primary"
                onClick={saveCalibration}
                disabled={draft.length !== 4 || !snapshot || Boolean(busy)}
              >
                Save calibration
              </button>
              {calibrated && (
                <button type="button" className="btn ghost" onClick={resetCalibration}>
                  Reset calibration
                </button>
              )}
            </div>
          </div>
        )}

        <label className="legacy-toggle">
          <input type="checkbox" checked={showLegacy} onChange={(e) => setShowLegacy(e.target.checked)} />
          Show legacy HSV ball detector output (debug only — not used for the Phase 1 baseline)
        </label>
        {showLegacy && (
          <p className="hint legacy-note">
            The old colour/contour detector is disabled on this screen on purpose. Its counts live on
            the <strong>Cameras</strong> tab under AI logs and are not part of the calibration
            baseline.
          </p>
        )}
      </section>

      {(busy || note || error) && (
        <div className={`banner ${error ? 'bad' : 'ok'}`}>{error || busy || note}</div>
      )}

      <section className="calib-split">
        <div className="panel calib-view">
          <div className="panel-head">
            <h2>Raw CCTV + ROI</h2>
            <span>{frameW && frameH ? `${frameW}×${frameH}` : '—'}</span>
          </div>
          {rawImage ? (
            <div className="calib-stage">
              <img src={rawImage} alt="Raw camera frame" className="calib-img" />
              <svg
                ref={svgRef}
                className="calib-overlay"
                viewBox={`0 0 ${frameW || 1} ${frameH || 1}`}
                preserveAspectRatio="xMidYMid meet"
                onClick={handleStageClick}
                onPointerMove={handlePointerMove}
                onPointerUp={handlePointerUp}
                onPointerLeave={handlePointerUp}
              >
                {shownCorners.length === 4 && (
                  <polygon
                    points={shownCorners.map((p) => `${p.x},${p.y}`).join(' ')}
                    className={`calib-poly ${draft.length === 4 ? 'draft' : ''}`}
                  />
                )}
                {shownCorners.length > 1 && shownCorners.length < 4 && (
                  <polyline
                    points={shownCorners.map((p) => `${p.x},${p.y}`).join(' ')}
                    className="calib-poly draft"
                    fill="none"
                  />
                )}
                {shownCorners.map((p, i) => (
                  <g key={i} className="calib-pt" onPointerDown={(e) => handlePointerDown(i, e)}>
                    <circle cx={p.x} cy={p.y} r={Math.max(6, (frameW || 960) / 130)} />
                    <text x={p.x + (frameW || 960) / 90} y={p.y - (frameW || 960) / 110}>
                      {CORNER_SHORT[i]}
                    </text>
                  </g>
                ))}
              </svg>
            </div>
          ) : (
            <div className="empty stage-empty">
              <div className="stage-frame" />
              <p>Press <strong>Grab frame</strong> to pull a still from the live camera.</p>
            </div>
          )}
        </div>

        <div className="panel calib-view">
          <div className="panel-head">
            <h2>Perspective-corrected table</h2>
            <span>{output ? `${output.output_width}×${output.output_height}` : '—'}</span>
          </div>
          {rightImage ? (
            <div className="calib-stage">
              <img src={rightImage} alt="Top-down table" className="calib-img" />
            </div>
          ) : (
            <div className="empty stage-empty">
              <div className="stage-frame" />
              <p>
                {editing
                  ? 'Place all four corners to see the top-down view.'
                  : 'Start live diagnostics to see the corrected table.'}
              </p>
            </div>
          )}
        </div>
      </section>

      <section className="panel calib-metrics">
        <div className="panel-head">
          <h2>Diagnostics</h2>
          <span className={calibrated ? 'tag ok' : 'tag'}>
            {active ? active.calibration_status : 'no table'}
          </span>
        </div>
        <dl className="metrics-grid">
          <div>
            <dt>Camera ID</dt>
            <dd title={cameraId}>{fmt(diag?.camera?.name || cameraId)}</dd>
          </div>
          <div>
            <dt>Table ID</dt>
            <dd title={active?.table_id}>{fmt(active?.name)}</dd>
          </div>
          <div>
            <dt>Camera resolution</dt>
            <dd>{fmt(diag?.source?.resolution || snapshot?.resolution)}</dd>
          </div>
          <div>
            <dt>Camera FPS</dt>
            <dd>{fmt(diag?.source?.camera_fps ?? snapshot?.camera_fps)}</dd>
          </div>
          <div>
            <dt>Capture FPS</dt>
            <dd>{fmt(diag?.source?.capture_fps ?? snapshot?.capture_fps)}</dd>
          </div>
          <div>
            <dt>Processing FPS</dt>
            <dd>{fmt(diag?.processing_fps)}</dd>
          </div>
          <div>
            <dt>Transform time</dt>
            <dd>{fmt(diag?.timings_ms?.transform ?? preview?.timings_ms?.transform, ' ms')}</dd>
          </div>
          <div>
            <dt>Frame grab</dt>
            <dd>{fmt(diag?.timings_ms?.grab, ' ms')}</dd>
          </div>
          <div>
            <dt>Round trip</dt>
            <dd>{fmt(diag?.timings_ms?.request, ' ms')}</dd>
          </div>
          <div>
            <dt>Output size</dt>
            <dd>{output ? `${output.output_width}×${output.output_height}` : '—'}</dd>
          </div>
          <div>
            <dt>Surface / margin</dt>
            <dd>
              {output ? `${output.surface_width}×${output.surface_height} · ${Math.round(output.margin * 100)}%` : '—'}
            </dd>
          </div>
          <div>
            <dt>Calibrated at</dt>
            <dd>
              {active?.calibration_timestamp
                ? new Date(active.calibration_timestamp).toLocaleString()
                : '—'}
            </dd>
          </div>
        </dl>
        {(diag?.warnings?.length || preview?.warnings?.length) > 0 && (
          <p className="hint warn">{(preview?.warnings || diag?.warnings).join(' ')}</p>
        )}
        {diag?.note && <p className="hint">{diag.note}</p>}
      </section>
    </div>
  );
}
