import express from 'express';
import cors from 'cors';
import http from 'http';
import { randomUUID } from 'crypto';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import { discoverOnvif, connectOnvifCamera, guessLocalHints } from './onvif.js';
import { scanNetworkForCameras, ssdpDiscover } from './network.js';
import { StreamManager } from './streamManager.js';
import { findWorkingRtsp, probeRtsp } from './rtsp.js';
import { fetchDahuaChannels, buildDahuaRtsp } from './dahua.js';
import { AiMonitor } from './aiMonitor.js';
import { createCalibrationRouter } from './calibrationRoutes.js';
import { createDatasetRouter } from './datasetRoutes.js';
import { createModelTestRouter } from './modelTestRoutes.js';
import { TableStore } from './tableStore.js';
import { FrameSource } from './frameSource.js';
import { ensureLocalCredentials, loadLocalCredentials, saveLocalCredentials } from './localCredentials.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
dotenv.config({ path: path.join(__dirname, '..', '.env') });

const app = express();
const server = http.createServer(app);
const streams = new StreamManager();
const aiMonitor = new AiMonitor();
streams.attachServer(server);

const PORT = process.env.PORT || 5050;
const cameras = new Map();

ensureLocalCredentials();

app.use(cors());
app.use(express.json({ limit: '4mb' }));

// Shared Phase 1 store + frame cache (calibration + dataset + model-test share one grabber).
const tableStore = new TableStore();
const frameSource = new FrameSource({ fps: Number(process.env.CALIB_FPS || 4) });

const cameraAccess = {
  getCamera: (id) => (id ? cameras.get(id) || null : null),
  listCameras: () => [...cameras.values()],
};

const calibrationRouter = createCalibrationRouter({
  ...cameraAccess,
  store: tableStore,
  frames: frameSource,
});
app.use('/api', calibrationRouter);

const datasetRouter = createDatasetRouter({
  ...cameraAccess,
  store: tableStore,
  frames: frameSource,
});
app.use('/api', datasetRouter);

const modelTestRouter = createModelTestRouter({
  ...cameraAccess,
  store: tableStore,
  frames: frameSource,
});
app.use('/api', modelTestRouter);

function findExisting({ id, ip, channel }) {
  if (id && cameras.has(id)) return cameras.get(id);
  return [...cameras.values()].find((c) => {
    if (channel != null) return c.ip === ip && Number(c.channel) === Number(channel);
    return c.ip === ip && (c.channel == null || c.isNvr);
  });
}

function upsertCamera(data) {
  const existing = findExisting({
    id: data.id,
    ip: data.ip,
    channel: data.channel,
  });
  const id = existing?.id || data.id || randomUUID();
  const camera = {
    id,
    name: data.name || existing?.name || `Camera ${data.ip}`,
    ip: data.ip,
    channel: data.channel ?? existing?.channel ?? null,
    isNvr: data.isNvr ?? existing?.isNvr ?? false,
    port: data.port ?? existing?.port ?? null,
    ports: data.ports || existing?.ports || [],
    protocol: data.protocol || existing?.protocol || 'unknown',
    brandHint: data.brandHint || existing?.brandHint || null,
    rtspUrl: data.rtspUrl !== undefined ? data.rtspUrl : existing?.rtspUrl || null,
    suggestedRtsp: data.suggestedRtsp || existing?.suggestedRtsp || null,
    manufacturer: data.manufacturer || existing?.manufacturer || null,
    model: data.model || existing?.model || null,
    source: data.source || existing?.source || 'manual',
    streams: data.streams || existing?.streams || [],
    connected: data.connected !== undefined ? data.connected : Boolean(existing?.connected),
    authError: data.authError !== undefined ? data.authError : existing?.authError || null,
    confidence: data.confidence || existing?.confidence || null,
    kind: data.kind || existing?.kind || null,
    reason: data.reason || existing?.reason || null,
    httpTitle: data.httpTitle || existing?.httpTitle || null,
    vlan: data.vlan || existing?.vlan || null,
    nvrIp: data.nvrIp || existing?.nvrIp || null,
    updatedAt: new Date().toISOString(),
  };
  cameras.set(id, camera);
  return camera;
}

function removeCamerasByIp(ip, { keepChannels = false } = {}) {
  for (const [id, cam] of cameras) {
    if (cam.ip !== ip && cam.nvrIp !== ip) continue;
    if (keepChannels && cam.channel != null) continue;
    cameras.delete(id);
  }
}

app.get('/api/health', (_req, res) => {
  res.json({ ok: true, time: new Date().toISOString() });
});

app.get('/api/credentials', (_req, res) => {
  res.json({ credentials: loadLocalCredentials() });
});

app.put('/api/credentials', (req, res) => {
  const saved = saveLocalCredentials({
    username: req.body?.username,
    password: req.body?.password,
  });
  res.json({ ok: true, credentials: saved });
});

app.get('/api/network', (_req, res) => {
  res.json({ interfaces: guessLocalHints() });
});

app.get('/api/cameras', (_req, res) => {
  const list = [...cameras.values()].sort((a, b) => {
    if (a.ip === b.ip) return (a.channel || 0) - (b.channel || 0);
    return String(a.ip).localeCompare(String(b.ip));
  });
  res.json({ cameras: list });
});

app.post('/api/cameras/discover', async (req, res) => {
  const { deepScan = true } = req.body || {};

  try {
    for (const [id, cam] of cameras) {
      if (cam.source !== 'manual' || !cam.connected) cameras.delete(id);
    }

    const [onvifDevices, ssdpDevices, scan] = await Promise.all([
      discoverOnvif(4500),
      ssdpDiscover(2500),
      deepScan
        ? scanNetworkForCameras()
        : Promise.resolve({ subnets: guessLocalHints(), cameras: [], stats: {} }),
    ]);

    for (const d of onvifDevices) {
      if (!d.ip) continue;
      upsertCamera({
        name: d.name || `ONVIF ${d.ip}`,
        ip: d.ip,
        port: d.port,
        protocol: 'onvif',
        source: 'onvif',
        connected: false,
        rtspUrl: null,
      });
    }

    for (const d of ssdpDevices) {
      upsertCamera({
        name: d.server || `SSDP ${d.ip}`,
        ip: d.ip,
        protocol: 'ssdp',
        source: 'ssdp',
        connected: false,
      });
    }

    for (const d of scan.cameras || []) {
      const isDahua = d.brandHint === 'dahua';
      upsertCamera({
        name: isDahua
          ? `Dahua NVR ${d.ip}`
          : d.brandHint === 'xmeye'
            ? `XMeye Device ${d.ip}`
            : `IP Camera ${d.ip}`,
        ip: d.ip,
        ports: d.ports,
        protocol: d.protocol,
        brandHint: d.brandHint,
        suggestedRtsp: d.suggestedRtsp,
        rtspUrl: null,
        connected: false,
        isNvr: isDahua,
        source: 'network-scan',
      });
    }

    const list = [...cameras.values()];
    const dahuaNvrs = list.filter((c) => c.brandHint === 'dahua');

    res.json({
      interfaces: scan.subnets || guessLocalHints(),
      discovered: {
        onvif: onvifDevices.length,
        ssdp: ssdpDevices.length,
        network: (scan.cameras || []).length,
      },
      cameras: list,
      tip:
        dahuaNvrs.length > 0
          ? 'Dahua NVR mila. Username/password daal kar "Load all channels" dabao — saari cams names ke sath aa jayengi.'
          : 'Username/password = NVR/camera login (WiFi password nahi).',
    });
  } catch (err) {
    res.status(500).json({ error: err.message || 'Discovery failed' });
  }
});

app.post('/api/cameras/:id/channels', async (req, res) => {
  const camera = cameras.get(req.params.id);
  if (!camera) return res.status(404).json({ error: 'Camera/NVR not found' });

  const { username = 'admin', password = '' } = req.body || {};
  if (!password && password !== '') {
    /* empty allowed */
  }

  try {
    const info = await fetchDahuaChannels({
      ip: camera.ip,
      username,
      password,
    });

    // Remove old NVR placeholder + previous channels for this IP
    for (const [id, cam] of cameras) {
      if (cam.ip === camera.ip || cam.nvrIp === camera.ip) cameras.delete(id);
    }

    const created = [];
    for (const ch of info.channels) {
      const rtspUrl = buildDahuaRtsp({
        ip: camera.ip,
        channel: ch.channel,
        username,
        password,
      });
      created.push(
        upsertCamera({
          name: ch.name,
          ip: camera.ip,
          channel: ch.channel,
          nvrIp: camera.ip,
          brandHint: 'dahua',
          protocol: 'rtsp',
          ports: camera.ports,
          rtspUrl,
          suggestedRtsp: `rtsp://${camera.ip}:554${ch.rtspPath}`,
          connected: true,
          isNvr: false,
          manufacturer: 'Dahua',
          model: info.deviceType,
          source: 'dahua-channels',
          authError: null,
        })
      );
    }

    res.json({
      deviceName: info.deviceName,
      deviceType: info.deviceType,
      count: created.length,
      cameras: [...cameras.values()].sort((a, b) => (a.channel || 0) - (b.channel || 0)),
      message: `${created.length} channels loaded from Dahua NVR${info.deviceName ? ` (${info.deviceName})` : ''}`,
    });
  } catch (err) {
    res.status(err.status === 401 ? 401 : 502).json({
      error: err.message || 'Failed to load channels',
      tip: 'Dahua NVR/web panel wala username password use karein.',
    });
  }
});

app.post('/api/cameras/manual', async (req, res) => {
  const { name, ip, rtspUrl, username, password } = req.body || {};
  if (!rtspUrl && !ip) {
    return res.status(400).json({ error: 'ip or rtspUrl required' });
  }

  const user = username || 'admin';
  const pass = password || '';

  try {
    const probe = await findWorkingRtsp({
      ip: ip || extractIp(rtspUrl),
      ports: [{ port: 554, protocol: 'rtsp' }],
      username: user,
      password: pass,
      preferredUrl: rtspUrl ? injectAuth(rtspUrl, user, pass) : null,
    });

    if (!probe.ok) {
      return res.status(401).json({
        error: probe.error,
        tip: 'Use the camera/NVR login (often admin + password from setup).',
      });
    }

    const camera = upsertCamera({
      name: name || `Manual ${ip || extractIp(probe.url)}`,
      ip: ip || extractIp(probe.url),
      rtspUrl: probe.url,
      protocol: 'rtsp',
      source: 'manual',
      connected: true,
      authError: null,
    });

    res.json({ camera });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post('/api/cameras/:id/connect', async (req, res) => {
  const camera = cameras.get(req.params.id);
  if (!camera) return res.status(404).json({ error: 'Camera not found' });

  const { username = 'admin', password = '', port, path, rtspUrl, loadChannels } = req.body || {};

  try {
    // Dahua NVR: expand all named channels
    if ((camera.brandHint === 'dahua' || camera.isNvr) && loadChannels !== false && !camera.channel && !rtspUrl) {
      const info = await fetchDahuaChannels({
        ip: camera.ip,
        username,
        password,
      });

      for (const [id, cam] of cameras) {
        if (cam.ip === camera.ip || cam.nvrIp === camera.ip) cameras.delete(id);
      }

      const created = info.channels.map((ch) =>
        upsertCamera({
          name: ch.name,
          ip: camera.ip,
          channel: ch.channel,
          nvrIp: camera.ip,
          brandHint: 'dahua',
          protocol: 'rtsp',
          ports: camera.ports,
          rtspUrl: buildDahuaRtsp({
            ip: camera.ip,
            channel: ch.channel,
            username,
            password,
          }),
          suggestedRtsp: `rtsp://${camera.ip}:554${ch.rtspPath}`,
          connected: true,
          manufacturer: 'Dahua',
          model: info.deviceType,
          source: 'dahua-channels',
        })
      );

      return res.json({
        cameras: created,
        camera: created[0] || null,
        message: `${created.length} cameras loaded with names from Dahua NVR`,
        expanded: true,
      });
    }

    if (rtspUrl) {
      const url = injectAuth(rtspUrl, username, password);
      const probe = await probeRtsp(url);
      if (!probe.ok) {
        camera.connected = false;
        camera.authError = probe.error;
        cameras.set(camera.id, camera);
        return res.status(401).json({ error: probe.error, camera });
      }
      camera.rtspUrl = url;
      camera.connected = true;
      camera.authError = null;
      cameras.set(camera.id, camera);
      return res.json({ camera, message: 'RTSP login OK' });
    }

    // Single channel already expanded
    if (camera.channel && camera.brandHint === 'dahua') {
      const url = buildDahuaRtsp({
        ip: camera.ip,
        channel: camera.channel,
        username,
        password,
      });
      const probe = await probeRtsp(url);
      if (!probe.ok) {
        camera.connected = false;
        camera.authError = probe.error;
        cameras.set(camera.id, camera);
        return res.status(401).json({ error: probe.error, camera });
      }
      camera.rtspUrl = url;
      camera.connected = true;
      camera.authError = null;
      cameras.set(camera.id, camera);
      return res.json({ camera, message: `Connected: ${camera.name}` });
    }

    let onvifUrl = null;
    try {
      const details = await connectOnvifCamera({
        ip: camera.ip,
        port: port || camera.port || 80,
        username,
        password,
        path,
      });
      camera.manufacturer = details.manufacturer;
      camera.model = details.model;
      camera.streams = details.streams;
      camera.name =
        [details.manufacturer, details.model].filter(Boolean).join(' ') || camera.name;
      onvifUrl = details.rtspUrl ? injectAuth(details.rtspUrl, username, password) : null;
    } catch {
      /* fall through */
    }

    const probe = await findWorkingRtsp({
      ip: camera.ip,
      ports: camera.ports,
      username,
      password,
      brandHint: camera.brandHint,
      preferredUrl:
        onvifUrl ||
        (camera.suggestedRtsp ? injectAuth(camera.suggestedRtsp, username, password) : null),
    });

    if (!probe.ok) {
      camera.connected = false;
      camera.rtspUrl = null;
      camera.authError = probe.error;
      cameras.set(camera.id, camera);
      return res.status(401).json({
        error: probe.error,
        tip: 'Camera/NVR username password daalein (WiFi nahi).',
        camera,
      });
    }

    camera.rtspUrl = probe.url;
    camera.connected = true;
    camera.authError = null;
    cameras.set(camera.id, camera);
    res.json({ camera, message: 'Connected — RTSP verified' });
  } catch (err) {
    res.status(err.status === 401 ? 401 : 502).json({
      error: err.message || 'Connect failed',
      tip: 'Dahua/NVR web login use karein.',
    });
  }
});

app.patch('/api/cameras/:id', async (req, res) => {
  const camera = cameras.get(req.params.id);
  if (!camera) return res.status(404).json({ error: 'Camera not found' });

  const { name, rtspUrl, username, password } = req.body || {};
  if (name) camera.name = name;
  if (rtspUrl) {
    const url = username ? injectAuth(rtspUrl, username, password || '') : rtspUrl;
    const probe = await probeRtsp(url);
    if (!probe.ok) {
      return res.status(401).json({ error: probe.error, camera });
    }
    camera.rtspUrl = url;
    camera.connected = true;
    camera.authError = null;
  }
  cameras.set(camera.id, camera);
  res.json({ camera });
});

app.delete('/api/cameras/:id', (req, res) => {
  const ok = cameras.delete(req.params.id);
  if (!ok) return res.status(404).json({ error: 'Camera not found' });
  for (const s of streams.list().filter((x) => x.cameraId === req.params.id)) {
    streams.stop(s.id);
  }
  res.json({ ok: true });
});

app.get('/api/streams', (_req, res) => {
  res.json({ streams: streams.list() });
});

app.post('/api/streams/start', async (req, res) => {
  const { cameraId, rtspUrl, name } = req.body || {};
  const camera = cameraId ? cameras.get(cameraId) : null;
  const url = rtspUrl || camera?.rtspUrl;

  if (!url) {
    return res.status(400).json({
      error: 'Pehle Connect / Load channels karein (sahi NVR username/password ke sath)',
    });
  }

  const probe = await probeRtsp(url);
  if (!probe.ok) {
    if (camera) {
      camera.connected = false;
      camera.authError = probe.error;
      cameras.set(camera.id, camera);
    }
    return res.status(401).json({ error: probe.error });
  }

  const stream = streams.start({
    cameraId: camera?.id || cameraId || 'manual',
    name: name || camera?.name || 'Live stream',
    rtspUrl: url,
  });

  res.json({
    stream: {
      id: stream.id,
      cameraId: stream.cameraId,
      name: stream.name,
      wsPath: `/stream/${stream.id}`,
      wsUrl: `ws://localhost:${PORT}/stream/${stream.id}`,
    },
  });
});

app.post('/api/streams/:id/stop', (req, res) => {
  const ok = streams.stop(req.params.id);
  if (!ok) return res.status(404).json({ error: 'Stream not found' });
  res.json({ ok: true });
});

app.get('/api/ai/health', async (_req, res) => {
  try {
    const r = await fetch(`${process.env.AI_WORKER_URL || 'http://127.0.0.1:5051'}/health`);
    const data = await r.json();
    res.json({ ok: r.ok, worker: data });
  } catch (err) {
    res.status(503).json({ ok: false, error: 'AI worker not reachable on :5051 — start ai-worker first' });
  }
});

app.get('/api/ai/jobs', (_req, res) => {
  res.json({ jobs: aiMonitor.listJobs() });
});

app.get('/api/ai/events', (req, res) => {
  const limit = Number(req.query.limit || 100);
  res.json({ events: aiMonitor.listEvents(limit) });
});

app.delete('/api/ai/events', (_req, res) => {
  aiMonitor.clearEvents();
  res.json({ ok: true, events: [] });
});

app.post('/api/ai/events/clear', (_req, res) => {
  aiMonitor.clearEvents();
  res.json({ ok: true, events: [] });
});

app.post('/api/ai/monitor/start', async (req, res) => {
  const { cameraId, rtspUrl, name } = req.body || {};
  const camera = cameraId ? cameras.get(cameraId) : null;
  const url = rtspUrl || camera?.rtspUrl;
  if (!url) {
    return res.status(400).json({
      error: 'Connected camera (with RTSP) required — pehle Live/Connect karke RTSP verify karein',
    });
  }

  try {
    // Quick worker ping
    const health = await fetch(`${process.env.AI_WORKER_URL || 'http://127.0.0.1:5051'}/health`);
    if (!health.ok) throw new Error('AI worker unhealthy');

    const job = await aiMonitor.start({
      cameraId: camera?.id || cameraId || 'manual',
      cameraName: name || camera?.name || 'Camera',
      rtspUrl: url,
    });

    res.json({
      job: {
        id: job.id,
        cameraId: job.cameraId,
        cameraName: job.cameraName,
        state: job.state,
      },
      message: 'AI monitor running — rack/frame events will appear in logs',
    });
  } catch (err) {
    res.status(503).json({
      error: err.message || 'Failed to start AI monitor',
      tip: 'Run: npm run dev:ai  (Python worker on port 5051)',
    });
  }
});

app.post('/api/ai/monitor/:id/stop', async (req, res) => {
  const ok = await aiMonitor.stop(req.params.id);
  // Idempotent: stale UI job ids after restart should not error
  res.json({ ok: true, alreadyStopped: !ok });
});

function extractIp(url) {
  try {
    return new URL(url).hostname;
  } catch {
    return null;
  }
}

function injectAuth(url, username, password) {
  if (!url) return url;
  if (!username) return url;
  try {
    const u = new URL(url);
    u.username = username;
    u.password = password || '';
    return u.toString();
  } catch {
    return url;
  }
}

server.listen(PORT, () => {
  console.log(`Camera backend listening on http://localhost:${PORT}`);
});

process.on('SIGINT', () => {
  aiMonitor.stopAll();
  streams.stopAll();
  calibrationRouter.stopAll();
  datasetRouter.stopAll();
  modelTestRouter.stopAll();
  frameSource.stopAll();
  process.exit(0);
});
