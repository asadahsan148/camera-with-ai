import { spawn } from 'child_process';
import { randomUUID } from 'crypto';
import fs from 'fs';
import os from 'os';
import path from 'path';

const AI_URL = process.env.AI_WORKER_URL || 'http://127.0.0.1:5051';
const SAMPLE_EVERY_MS = Number(process.env.AI_SAMPLE_MS || 1500);

export class AiMonitor {
  constructor() {
    this.jobs = new Map();
    this.events = [];
    this.maxEvents = 500;
  }

  listJobs() {
    return [...this.jobs.values()].map((j) => ({
      id: j.id,
      cameraId: j.cameraId,
      cameraName: j.cameraName,
      rtspUrl: mask(j.rtspUrl),
      running: j.running,
      state: j.state,
      lastAnalysis: j.lastAnalysis,
      startedAt: j.startedAt,
      error: j.error,
      samples: j.samples,
    }));
  }

  listEvents(limit = 100) {
    return this.events.slice(0, limit);
  }

  clearEvents() {
    this.events = [];
  }

  pushEvent(event) {
    const row = {
      id: randomUUID(),
      ts: new Date().toISOString(),
      ...event,
    };
    this.events.unshift(row);
    if (this.events.length > this.maxEvents) this.events.length = this.maxEvents;
    console.log(
      `[AI] ${row.ts} ${row.type} camera=${row.cameraName || row.cameraId} ${row.message || ''}`
    );
    return row;
  }

  async start({ cameraId, cameraName, rtspUrl }) {
    if (!rtspUrl) throw new Error('rtspUrl required for AI monitor');

    const existing = [...this.jobs.values()].find(
      (j) => j.cameraId === cameraId && j.running
    );
    if (existing) return existing;

    const id = randomUUID();
    const job = {
      id,
      cameraId,
      cameraName: cameraName || cameraId,
      rtspUrl: preferAiStream(rtspUrl),
      running: true,
      state: 'STARTING',
      lastAnalysis: null,
      startedAt: new Date().toISOString(),
      error: null,
      samples: 0,
      timer: null,
      busy: false,
    };

    this.jobs.set(id, job);
    this.pushEvent({
      type: 'MONITOR_STARTED',
      cameraId,
      cameraName: job.cameraName,
      jobId: id,
      message: 'AI monitor started',
    });

    try {
      await fetch(`${AI_URL}/sessions/${id}/reset`, { method: 'POST' });
    } catch {
      /* worker may start slightly later */
    }

    const tick = async () => {
      if (!job.running || job.busy) return;
      job.busy = true;
      try {
        const { jpeg, stream } = await grabBestJpegFrame(job.rtspUrl);
        const result = await analyzeJpeg(jpeg, id);
        job.samples += 1;
        job.state = result.state || job.state;
        job.lastAnalysis = result.analysis || null;
        if (job.lastAnalysis && typeof job.lastAnalysis === 'object') {
          job.lastAnalysis.aiStream = stream;
        }
        job.error = null;

        for (const ev of result.events || []) {
          this.pushEvent({
            ...ev,
            cameraId,
            cameraName: job.cameraName,
            jobId: id,
            analysis: result.analysis,
          });
        }
      } catch (err) {
        job.error = err.message;
        if (job.samples === 0 || job.samples % 10 === 0) {
          this.pushEvent({
            type: 'MONITOR_ERROR',
            cameraId,
            cameraName: job.cameraName,
            jobId: id,
            message: err.message,
          });
        }
      } finally {
        job.busy = false;
      }
    };

    setTimeout(tick, 500);
    job.timer = setInterval(tick, SAMPLE_EVERY_MS);
    return job;
  }

  async stop(jobId) {
    const job = this.jobs.get(jobId);
    if (!job) return false;
    job.running = false;
    if (job.timer) clearInterval(job.timer);
    this.jobs.delete(jobId);
    try {
      await fetch(`${AI_URL}/sessions/${jobId}`, { method: 'DELETE' });
    } catch {
      /* ignore */
    }
    this.pushEvent({
      type: 'MONITOR_STOPPED',
      cameraId: job.cameraId,
      cameraName: job.cameraName,
      jobId,
      message: 'AI monitor stopped',
    });
    return true;
  }

  stopAll() {
    for (const id of [...this.jobs.keys()]) {
      this.stop(id);
    }
  }
}

function preferAiStream(rtspUrl) {
  // Main stream (subtype=0) — better ball resolution for counting
  try {
    if (/subtype=\d+/i.test(rtspUrl)) {
      return rtspUrl.replace(/subtype=\d+/i, 'subtype=0');
    }
    if (rtspUrl.includes('?')) return `${rtspUrl}&subtype=0`;
    return `${rtspUrl}?subtype=0`;
  } catch {
    return rtspUrl;
  }
}

function preferSubstream(rtspUrl) {
  try {
    if (/subtype=\d+/i.test(rtspUrl)) {
      return rtspUrl.replace(/subtype=\d+/i, 'subtype=1');
    }
    if (rtspUrl.includes('?')) return `${rtspUrl}&subtype=1`;
    return `${rtspUrl}?subtype=1`;
  } catch {
    return rtspUrl;
  }
}

function isUsableJpeg(buf) {
  // Black / failed NVR grabs are tiny; good table frames are usually >> 8KB
  if (!buf || buf.length < 8000) return false;
  return true;
}

async function grabBestJpegFrame(rtspUrl) {
  const mainUrl = preferAiStream(rtspUrl);
  const subUrl = preferSubstream(rtspUrl);
  let mainErr = null;
  try {
    const jpeg = await grabJpegFrame(mainUrl);
    if (isUsableJpeg(jpeg)) return { jpeg, stream: 'main' };
  } catch (e) {
    mainErr = e;
  }
  try {
    const jpeg = await grabJpegFrame(subUrl);
    if (isUsableJpeg(jpeg)) return { jpeg, stream: 'sub' };
    // Even a small sub frame is better than nothing if main failed
    if (jpeg && jpeg.length > 1500) return { jpeg, stream: 'sub-weak' };
  } catch (e) {
    if (mainErr) throw mainErr;
    throw e;
  }
  throw mainErr || new Error('No usable camera frame (main+sub)');
}

function mask(url) {
  try {
    const u = new URL(url);
    if (u.password) u.password = '****';
    return u.toString();
  } catch {
    return String(url).replace(/:([^:@]+)@/, ':****@');
  }
}

function grabJpegFrame(rtspUrl) {
  return new Promise((resolve, reject) => {
    const tmp = path.join(os.tmpdir(), `snooker-ai-${randomUUID()}.jpg`);
    const args = [
      '-hide_banner',
      '-loglevel',
      'error',
      '-rtsp_transport',
      'tcp',
      '-i',
      rtspUrl,
      '-frames:v',
      '1',
      '-q:v',
      '3',
      '-y',
      tmp,
    ];
    const ff = spawn('ffmpeg', args, { windowsHide: true });
    let err = '';
    ff.stderr.on('data', (d) => {
      err += d.toString();
    });
    ff.on('close', (code) => {
      if (code !== 0) {
        try {
          fs.unlinkSync(tmp);
        } catch {
          /* ignore */
        }
        reject(new Error(err.trim() || `ffmpeg frame grab failed (${code})`));
        return;
      }
      try {
        const buf = fs.readFileSync(tmp);
        fs.unlinkSync(tmp);
        resolve(buf);
      } catch (e) {
        reject(e);
      }
    });
  });
}

async function analyzeJpeg(jpegBuffer, sessionId) {
  const form = new FormData();
  form.append('session_id', sessionId);
  form.append('file', new Blob([jpegBuffer], { type: 'image/jpeg' }), 'frame.jpg');

  const res = await fetch(`${AI_URL}/analyze`, {
    method: 'POST',
    body: form,
  });

  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = Array.isArray(data.detail)
      ? data.detail.map((d) => d.msg || d).join('; ')
      : data.detail || data.error;
    throw new Error(detail || `AI worker HTTP ${res.status}`);
  }
  return data;
}
