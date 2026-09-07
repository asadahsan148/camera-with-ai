import { useEffect, useState } from 'react';
import { api } from './api';
import LivePlayer from './LivePlayer';
import Calibration from './Calibration';
import VisionDataset from './VisionDataset';
import VisionModelTest from './VisionModelTest';
import './App.css';

function pathToView(pathname) {
  const p = (pathname || '/').replace(/\/+$/, '') || '/';
  if (p === '/vision-model-test' || p.endsWith('/vision-model-test')) return 'model-test';
  if (p === '/vision-dataset' || p.endsWith('/vision-dataset')) return 'dataset';
  if (p === '/table-calibration' || p.endsWith('/table-calibration')) return 'calibration';
  return 'cameras';
}

function viewToPath(view) {
  if (view === 'model-test') return '/vision-model-test';
  if (view === 'dataset') return '/vision-dataset';
  if (view === 'calibration') return '/table-calibration';
  return '/';
}

export default function App() {
  const [cameras, setCameras] = useState([]);
  const [interfaces, setInterfaces] = useState([]);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [activeStream, setActiveStream] = useState(null);
  const [creds, setCreds] = useState({ username: 'admin', password: 'Raja7860' });
  const [manual, setManual] = useState({
    name: '',
    ip: '',
    rtspUrl: '',
    username: 'admin',
    password: 'Raja7860',
  });
  const [selectedId, setSelectedId] = useState(null);
  const [rtspEdit, setRtspEdit] = useState('');
  const [aiEvents, setAiEvents] = useState([]);
  const [aiJobs, setAiJobs] = useState([]);
  const [aiOnline, setAiOnline] = useState(false);
  const [view, setView] = useState(() => pathToView(window.location.pathname));
  const [legacyDebug, setLegacyDebug] = useState(false);

  function go(next) {
    const path = viewToPath(next);
    if (window.location.pathname !== path) {
      window.history.pushState({}, '', path);
    }
    setView(next);
  }

  useEffect(() => {
    const onPop = () => setView(pathToView(window.location.pathname));
    window.addEventListener('popstate', onPop);
    return () => window.removeEventListener('popstate', onPop);
  }, []);

  useEffect(() => {
    api
      .getCredentials()
      .then((d) => {
        const c = d.credentials || {};
        const next = {
          username: c.username || 'admin',
          password: c.password || 'Raja7860',
        };
        setCreds(next);
        setManual((m) => ({ ...m, username: next.username, password: next.password }));
      })
      .catch(() => {});
    api
      .network()
      .then((d) => setInterfaces(d.interfaces || []))
      .catch(() => {});
    api
      .listCameras()
      .then((d) => setCameras(d.cameras || []))
      .catch(() => {});
  }, []);

  function updateCreds(patch) {
    setCreds((prev) => {
      const next = { ...prev, ...patch };
      api.saveCredentials(next).catch(() => {});
      setManual((m) => ({
        ...m,
        username: next.username,
        password: next.password,
      }));
      return next;
    });
  }

  useEffect(() => {
    let cancelled = false;
    async function pollAi() {
      try {
        const [health, ev, jobs] = await Promise.all([
          api.aiHealth().catch(() => ({ ok: false })),
          api.aiEvents(80).catch(() => ({ events: [] })),
          api.aiJobs().catch(() => ({ jobs: [] })),
        ]);
        if (cancelled) return;
        setAiOnline(Boolean(health.ok));
        setAiEvents(ev.events || []);
        setAiJobs(jobs.jobs || []);
      } catch {
        if (!cancelled) setAiOnline(false);
      }
    }
    pollAi();
    const t = setInterval(pollAi, 2000);
    return () => {
      cancelled = true;
      clearInterval(t);
    };
  }, []);

  const selected = cameras.find((c) => c.id === selectedId) || null;

  async function discover() {
    setLoading(true);
    setError('');
    setMessage('Scanning your WiFi subnet for cameras / NVR…');
    try {
      const data = await api.discover(true);
      setCameras(data.cameras || []);
      setInterfaces(data.interfaces || []);
      const d = data.discovered || {};
      setMessage(
        data.tip ||
          `Found ${data.cameras?.length || 0} device(s) — ONVIF ${d.onvif || 0}, SSDP ${d.ssdp || 0}, port scan ${d.network || 0}`
      );
    } catch (err) {
      setError(err.message);
      setMessage('');
    } finally {
      setLoading(false);
    }
  }

  async function loadAllChannels(camera) {
    setLoading(true);
    setError('');
    setMessage(`Loading all channels from ${camera.ip} (Digest login)…`);
    try {
      const data = await api.loadChannels(camera.id, {
        username: creds.username || 'admin',
        password: creds.password,
      });
      setCameras(data.cameras || []);
      if (data.cameras?.[0]) {
        setSelectedId(data.cameras[0].id);
        setRtspEdit(data.cameras[0].rtspUrl || '');
      }
      setMessage(data.message || `${data.count} cameras loaded`);
    } catch (err) {
      setError(err.message);
      setMessage('');
    } finally {
      setLoading(false);
    }
  }

  async function connectCamera(camera) {
    if (!creds.password) {
      setError('Pehle camera/NVR password daalein (WiFi password nahi).');
      return;
    }
    setLoading(true);
    setError('');
    setMessage(`Connecting to ${camera.name || camera.ip}…`);
    try {
      const data = await api.connect(camera.id, {
        username: creds.username || 'admin',
        password: creds.password,
        rtspUrl: camera.channel ? undefined : rtspEdit || undefined,
        loadChannels: Boolean(camera.isNvr || (camera.brandHint === 'dahua' && !camera.channel)),
      });
      if (data.expanded && data.cameras) {
        setCameras(data.cameras);
        setSelectedId(data.cameras[0]?.id || null);
        setRtspEdit(data.cameras[0]?.rtspUrl || '');
      } else {
        setCameras((prev) => prev.map((c) => (c.id === camera.id ? data.camera : c)));
        setSelectedId(camera.id);
        setRtspEdit(data.camera?.rtspUrl || '');
      }
      setMessage(data.message || `Connected: ${data.camera?.name || camera.name}`);
    } catch (err) {
      setError(err.message);
      setMessage('');
    } finally {
      setLoading(false);
    }
  }

  async function startLive(camera) {
    setLoading(true);
    setError('');
    try {
      let cam = camera;
      if (!cam.connected || !cam.rtspUrl) {
        if (!creds.password) {
          throw new Error('Pehle camera username/password daal kar Connect karein.');
        }
        const data = await api.connect(cam.id, {
          username: creds.username || 'admin',
          password: creds.password,
          rtspUrl: rtspEdit || undefined,
        });
        cam = data.camera;
        setCameras((prev) => prev.map((c) => (c.id === cam.id ? cam : c)));
        setRtspEdit(cam.rtspUrl || '');
      }

      const data = await api.startStream({
        cameraId: cam.id,
        rtspUrl: cam.rtspUrl,
        name: cam.name,
      });

      const wsUrl =
        `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.hostname}:5050${data.stream.wsPath}`;

      setActiveStream({
        ...data.stream,
        wsUrl,
        title: cam.name,
      });
      setMessage(`Streaming ${cam.name}`);
    } catch (err) {
      setError(err.message);
      setMessage('');
    } finally {
      setLoading(false);
    }
  }

  async function stopLive() {
    if (!activeStream) return;
    try {
      await api.stopStream(activeStream.id);
    } catch {
      /* ignore */
    }
    setActiveStream(null);
  }

  async function startAi(camera) {
    setLoading(true);
    setError('');
    try {
      let cam = camera;
      if (!cam.rtspUrl) {
        throw new Error('Pehle camera Connect karein taake RTSP ready ho');
      }
      const data = await api.aiStart({
        cameraId: cam.id,
        rtspUrl: cam.rtspUrl,
        name: cam.name,
      });
      setMessage(data.message || 'AI monitor started');
      const jobs = await api.aiJobs();
      setAiJobs(jobs.jobs || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function stopAi(jobId) {
    try {
      await api.aiStop(jobId);
    } catch {
      /* stale job after backend restart — still refresh list */
    }
    try {
      const jobs = await api.aiJobs();
      setAiJobs(jobs.jobs || []);
      setMessage('AI monitor stopped');
    } catch (err) {
      setError(err.message);
    }
  }

  async function clearAiLogs() {
    try {
      await api.aiClearEvents();
      setAiEvents([]);
      setMessage('AI logs cleared');
    } catch (err) {
      setError(err.message);
    }
  }

  async function addManualCamera(e) {
    e.preventDefault();
    setLoading(true);
    setError('');
    try {
      const data = await api.addManual(manual);
      setCameras((prev) => {
        const others = prev.filter((c) => c.id !== data.camera.id);
        return [...others, data.camera];
      });
      setSelectedId(data.camera.id);
      setRtspEdit(data.camera.rtspUrl || '');
      setMessage('Manual camera added');
      setManual((m) => ({ ...m, name: '', ip: '', rtspUrl: '' }));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <header className="hero">
        <div className="brand">
          <span className="brand-mark" />
          <div>
            <p className="eyebrow">Local WiFi</p>
            <h1>WiFi Cameras</h1>
          </div>
        </div>
        <p className="lede">
          Discover IP cameras on your connected network, authenticate, and watch a low-latency live RTSP stream in the browser.
        </p>
        <p className="cred-tip">
          Username / password = <strong>NVR/camera login</strong> (Dahua web panel / DMSS app). WiFi password nahi.
          Agar 1 device dikhe to woh NVR hai — login ke baad <strong>Load all channels</strong> se saari cams names ke sath aati hain.
        </p>
        <div className="hero-actions">
          <button type="button" className="btn primary" onClick={discover} disabled={loading}>
            {loading ? 'Scanning…' : 'Scan WiFi cameras'}
          </button>
          <div className="creds">
            <input
              placeholder="Camera username"
              value={creds.username}
              onChange={(e) => updateCreds({ username: e.target.value })}
            />
            <input
              placeholder="NVR password (saved locally)"
              type="password"
              value={creds.password}
              onChange={(e) => updateCreds({ password: e.target.value })}
              autoComplete="off"
            />
          </div>
        </div>
        {interfaces.length > 0 && (
          <div className="ifaces">
            {interfaces.map((iface) => (
              <span key={`${iface.interface}-${iface.ip}`}>
                {iface.interface}: {iface.ip} ({iface.subnet})
              </span>
            ))}
          </div>
        )}
        <nav className="tabs">
          <button
            type="button"
            className={view === 'cameras' ? 'tab on' : 'tab'}
            onClick={() => go('cameras')}
          >
            Cameras &amp; live
          </button>
          <button
            type="button"
            className={view === 'calibration' ? 'tab on' : 'tab'}
            onClick={() => go('calibration')}
          >
            Table calibration
          </button>
          <button
            type="button"
            className={view === 'dataset' ? 'tab on' : 'tab'}
            onClick={() => go('dataset')}
          >
            Vision dataset
          </button>
          <button
            type="button"
            className={view === 'model-test' ? 'tab on' : 'tab'}
            onClick={() => go('model-test')}
          >
            Model test
          </button>
        </nav>
      </header>

      {(message || error) && (
        <div className={`banner ${error ? 'bad' : 'ok'}`}>{error || message}</div>
      )}

      {view === 'calibration' && <Calibration cameras={cameras} />}
      {view === 'dataset' && <VisionDataset cameras={cameras} />}
      {view === 'model-test' && <VisionModelTest cameras={cameras} />}

      <main className="layout" hidden={view !== 'cameras'}>
        <section className="panel list-panel">
          <div className="panel-head">
            <h2>Cameras</h2>
            <span>{cameras.length}</span>
          </div>

          {cameras.length === 0 ? (
            <div className="empty">
              <p>
                No cameras yet. Scan WiFi — agar Dahua NVR mile to password daal kar <strong>Load all channels</strong> se
                saari cams names ke sath load hongi.
              </p>
            </div>
          ) : (
            <ul className="camera-list">
              {cameras.map((cam) => (
                <li
                  key={cam.id}
                  className={selectedId === cam.id ? 'active' : ''}
                  onClick={() => {
                    setSelectedId(cam.id);
                    setRtspEdit(cam.rtspUrl || '');
                  }}
                >
                  <div>
                    <strong>{cam.name}</strong>
                    <small>
                      {cam.ip}
                      {cam.channel != null ? ` · CH ${cam.channel}` : ''}
                      {cam.manufacturer ? ` · ${cam.manufacturer}` : ''}
                      {cam.model ? ` ${cam.model}` : ''}
                    </small>
                  </div>
                  <div className="tags">
                    <span className="tag">{cam.channel != null ? `ch-${cam.channel}` : cam.brandHint || cam.source}</span>
                    {cam.isNvr && <span className="tag">NVR</span>}
                    {cam.connected ? (
                      <span className="tag ok">verified</span>
                    ) : (
                      <span className="tag">needs login</span>
                    )}
                  </div>
                  <div className="row-actions" onClick={(e) => e.stopPropagation()}>
                    {(cam.isNvr || (cam.brandHint === 'dahua' && cam.channel == null)) && (
                      <button
                        type="button"
                        className="btn primary"
                        onClick={() => loadAllChannels(cam)}
                        disabled={loading}
                      >
                        Load all channels
                      </button>
                    )}
                    {cam.channel != null && (
                      <>
                        <button type="button" className="btn primary" onClick={() => startLive(cam)} disabled={loading}>
                          Live
                        </button>
                        {legacyDebug && (
                          <button
                            type="button"
                            className="btn ghost"
                            onClick={() => startAi(cam)}
                            disabled={loading || !cam.rtspUrl}
                            title={!cam.rtspUrl ? 'Connect/verify RTSP first' : 'Legacy HSV detector'}
                          >
                            Legacy AI
                          </button>
                        )}
                      </>
                    )}
                    {!cam.isNvr && cam.channel == null && (
                      <>
                        <button type="button" className="btn ghost" onClick={() => connectCamera(cam)} disabled={loading}>
                          Connect
                        </button>
                        <button type="button" className="btn primary" onClick={() => startLive(cam)} disabled={loading}>
                          Live
                        </button>
                        {legacyDebug && (
                          <button
                            type="button"
                            className="btn ghost"
                            onClick={() => startAi(cam)}
                            disabled={loading || !cam.rtspUrl}
                          >
                            Legacy AI
                          </button>
                        )}
                      </>
                    )}
                  </div>
                </li>
              ))}
            </ul>
          )}

          <form className="manual" onSubmit={addManualCamera}>
            <h3>Add manually</h3>
            <input
              placeholder="Name"
              value={manual.name}
              onChange={(e) => setManual((m) => ({ ...m, name: e.target.value }))}
            />
            <input
              placeholder="IP address"
              value={manual.ip}
              onChange={(e) => setManual((m) => ({ ...m, ip: e.target.value }))}
            />
            <input
              placeholder="rtsp://user:pass@ip:554/path"
              value={manual.rtspUrl}
              onChange={(e) => setManual((m) => ({ ...m, rtspUrl: e.target.value }))}
            />
            <div className="creds">
              <input
                placeholder="Username"
                value={manual.username}
                onChange={(e) => setManual((m) => ({ ...m, username: e.target.value }))}
              />
              <input
                placeholder="Password"
                type="password"
                value={manual.password}
                onChange={(e) => setManual((m) => ({ ...m, password: e.target.value }))}
              />
            </div>
            <button type="submit" className="btn ghost" disabled={loading}>
              Save camera
            </button>
          </form>
        </section>

        <div className="watch-split">
          <section className="panel view-panel">
            <div className="panel-head">
              <h2>Live view</h2>
            </div>

            {activeStream ? (
              <LivePlayer
                key={activeStream.id}
                wsUrl={activeStream.wsUrl}
                title={activeStream.title}
                onStop={stopLive}
              />
            ) : (
              <div className="empty stage-empty">
                <div className="stage-frame" />
                <p>Select a camera and press Live to start streaming.</p>
              </div>
            )}

            {selected && (
              <div className="details">
                <h3>{selected.name}</h3>
                <dl>
                  <div>
                    <dt>IP</dt>
                    <dd>{selected.ip}</dd>
                  </div>
                  <div>
                    <dt>Source</dt>
                    <dd>{selected.source}</dd>
                  </div>
                  <div>
                    <dt>Protocol</dt>
                    <dd>{selected.protocol}</dd>
                  </div>
                </dl>
                <label>
                  RTSP URL
                  <input
                    value={rtspEdit}
                    onChange={(e) => setRtspEdit(e.target.value)}
                    placeholder="rtsp://admin:password@192.168.x.x:554/Streaming/Channels/101"
                  />
                </label>
                <p className="hint">
                  Common paths: Hikvision <code>/Streaming/Channels/101</code>, Dahua{' '}
                  <code>/cam/realmonitor?channel=1&subtype=0</code>, generic <code>/stream1</code>
                </p>
              </div>
            )}
          </section>

          <section className="panel ai-panel">
            <div className="panel-head">
              <h2>Legacy detector (debug)</h2>
              <div className="ai-head-actions">
                <span className={aiOnline ? 'tag ok' : 'tag'}>
                  {aiOnline ? 'online' : 'offline'}
                </span>
                <button
                  type="button"
                  className="btn ghost"
                  disabled={aiEvents.length === 0 || !legacyDebug}
                  onClick={clearAiLogs}
                >
                  Clear
                </button>
              </div>
            </div>
            <label className="legacy-toggle">
              <input
                type="checkbox"
                checked={legacyDebug}
                onChange={(e) => setLegacyDebug(e.target.checked)}
              />
              Enable legacy HSV ball detector output
            </label>
            <p className="hint ai-hint">
              The colour/contour ball counter is <strong>not</strong> the Phase 1 baseline. Use{' '}
              <strong>Table calibration</strong> for the clean geometry view; this panel stays off
              unless you switch it on for debugging.
            </p>

            {legacyDebug && aiJobs.length > 0 && (
              <div className="ai-jobs">
                {aiJobs.map((j) => (
                  <div key={j.id} className="ai-job">
                    <div>
                      <strong>{j.cameraName}</strong>
                      <small>
                        {j.state}
                        {j.lastAnalysis
                          ? ` · ${j.lastAnalysis.table_label || 'near'} · ${j.lastAnalysis.phase || '-'}${
                              j.lastAnalysis.detector_ran === false ? ' · skip' : ''
                            } · balls=${j.lastAnalysis.detections ?? '-'} · reds=${j.lastAnalysis.counts?.red ?? '-'}${
                              j.lastAnalysis.count_frozen ? ' · ACTIVE' : ''
                            }${
                              j.lastAnalysis.counts
                                ? ` · Y${j.lastAnalysis.counts.yellow || 0}/G${j.lastAnalysis.counts.green || 0}/Br${j.lastAnalysis.counts.brown || 0}/Bl${j.lastAnalysis.counts.blue || 0}/P${j.lastAnalysis.counts.pink || 0}/K${j.lastAnalysis.counts.black || 0}/W${j.lastAnalysis.counts.white || 0}`
                                : ''
                            }`
                          : ` · samples=${j.samples}`}
                        {j.error ? ` · ${j.error}` : ''}
                      </small>
                    </div>
                    <button type="button" className="btn ghost" onClick={() => stopAi(j.id)}>
                      Stop
                    </button>
                  </div>
                ))}
              </div>
            )}

            <ul className="ai-events" hidden={!legacyDebug}>
              {aiEvents.length === 0 ? (
                <li className="empty">No legacy detector events.</li>
              ) : (
                aiEvents.map((ev) => (
                  <li key={ev.id} className={`ai-ev ${ev.type}`}>
                    <time>{new Date(ev.ts).toLocaleTimeString()}</time>
                    <strong>{ev.type}</strong>
                    <span>
                      {ev.cameraName || ev.cameraId}
                      {ev.table_label ? ` · ${ev.table_label}` : ''}
                      {ev.frame_id != null ? ` · #${ev.frame_id}` : ''}
                      {ev.message ? ` — ${ev.message}` : ''}
                    </span>
                  </li>
                ))
              )}
            </ul>
          </section>
        </div>
      </main>
    </div>
  );
}
