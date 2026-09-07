const API = '/api';

async function request(path, options = {}) {
  const res = await fetch(`${API}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const tip = data.tip ? ` — ${data.tip}` : '';
    throw new Error((data.error || res.statusText || 'Request failed') + tip);
  }
  return data;
}

export const api = {
  health: () => request('/health'),
  network: () => request('/network'),
  getCredentials: () => request('/credentials'),
  saveCredentials: (payload) =>
    request('/credentials', { method: 'PUT', body: JSON.stringify(payload) }),
  listCameras: () => request('/cameras'),
  discover: (deepScan = true) =>
    request('/cameras/discover', {
      method: 'POST',
      body: JSON.stringify({ deepScan }),
    }),
  addManual: (payload) =>
    request('/cameras/manual', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  loadChannels: (id, payload) =>
    request(`/cameras/${id}/channels`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  connect: (id, payload) =>
    request(`/cameras/${id}/connect`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  updateCamera: (id, payload) =>
    request(`/cameras/${id}`, {
      method: 'PATCH',
      body: JSON.stringify(payload),
    }),
  startStream: (payload) =>
    request('/streams/start', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  stopStream: (id) =>
    request(`/streams/${id}/stop`, {
      method: 'POST',
    }),
  listStreams: () => request('/streams'),
  aiHealth: () => request('/ai/health'),
  aiJobs: () => request('/ai/jobs'),
  aiEvents: (limit = 100) => request(`/ai/events?limit=${limit}`),
  aiClearEvents: () =>
    request('/ai/events/clear', {
      method: 'POST',
      body: JSON.stringify({}),
    }),
  aiStart: (payload) =>
    request('/ai/monitor/start', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  aiStop: (id) =>
    request(`/ai/monitor/${id}/stop`, {
      method: 'POST',
    }),

  // --- Phase 1: table calibration & diagnostics ---
  listTables: (cameraId) =>
    request(`/tables${cameraId ? `?cameraId=${encodeURIComponent(cameraId)}` : ''}`),
  createTable: (payload) =>
    request('/tables', { method: 'POST', body: JSON.stringify(payload) }),
  updateTable: (tableId, payload) =>
    request(`/tables/${tableId}`, { method: 'PATCH', body: JSON.stringify(payload) }),
  deleteTable: (tableId) => request(`/tables/${tableId}`, { method: 'DELETE' }),
  calibrateTable: (tableId, payload) =>
    request(`/tables/${tableId}/calibrate`, { method: 'POST', body: JSON.stringify(payload) }),
  resetTable: (tableId) => request(`/tables/${tableId}/reset`, { method: 'POST' }),
  previewTable: (tableId, payload) =>
    request(`/tables/${tableId}/preview`, { method: 'POST', body: JSON.stringify(payload) }),
  snapshot: (cameraId) => request(`/cameras/${cameraId}/snapshot`),
  diagnostics: (tableId, { overlay = true } = {}) =>
    request(`/tables/${tableId}/diagnostics?overlay=${overlay}`),

  // --- Phase 2: vision dataset ---
  datasetStats: () => request('/dataset/stats'),
  datasetClasses: () => request('/dataset/classes'),
  datasetImages: (params = {}) => {
    const q = new URLSearchParams();
    if (params.annotated === true) q.set('annotated', 'true');
    if (params.annotated === false) q.set('annotated', 'false');
    if (params.limit != null) q.set('limit', String(params.limit));
    if (params.offset != null) q.set('offset', String(params.offset));
    const qs = q.toString();
    return request(`/dataset/images${qs ? `?${qs}` : ''}`);
  },
  datasetImage: (imageId) => request(`/dataset/images/${imageId}`),
  datasetDeleteImage: (imageId) => request(`/dataset/images/${imageId}`, { method: 'DELETE' }),
  datasetPreview: (tableId) => request(`/dataset/preview/${tableId}`),
  datasetCapture: (payload) =>
    request('/dataset/capture', { method: 'POST', body: JSON.stringify(payload) }),
  datasetGetAnnotation: (imageId) => request(`/dataset/images/${imageId}/annotation`),
  datasetSaveAnnotation: (imageId, payload) =>
    request(`/dataset/images/${imageId}/annotation`, {
      method: 'PUT',
      body: JSON.stringify(payload),
    }),
  datasetExport: () => request('/dataset/export', { method: 'POST', body: JSON.stringify({}) }),
  datasetFileUrl: (imageId) => `/api/dataset/images/${imageId}/file`,

  // --- Phase 2B: pretrained model benchmark ---
  modelTestStatus: () => request('/model-test/status'),
  modelTestDetectors: () => request('/model-test/detectors'),
  modelTestInfer: (payload) =>
    request('/model-test/infer', { method: 'POST', body: JSON.stringify(payload) }),
  modelTestSnapshot: (payload) =>
    request('/model-test/snapshot', { method: 'POST', body: JSON.stringify(payload) }),
  modelTestSnapshots: () => request('/model-test/snapshots'),
};
