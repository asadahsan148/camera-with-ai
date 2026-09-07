/**
 * Low-rate MJPEG frame source used by the table calibration / diagnostics
 * screen.
 *
 * Deliberately separate from StreamManager (browser mpegts live view) and from
 * aiMonitor (legacy detector sampling) so calibration never disturbs the
 * working camera streaming path. It only consumes an RTSP URL that the
 * existing camera code already produced.
 */

import { spawn } from 'child_process';

const SOI = Buffer.from([0xff, 0xd8, 0xff]);
const EOI = Buffer.from([0xff, 0xd9]);
const IDLE_STOP_MS = 60_000;
const MAX_BUFFER = 12 * 1024 * 1024;

/** Read width/height straight out of the JPEG SOFn marker. */
export function jpegSize(buf) {
  if (!buf || buf.length < 4 || buf[0] !== 0xff || buf[1] !== 0xd8) return null;
  let i = 2;
  while (i < buf.length - 9) {
    if (buf[i] !== 0xff) {
      i += 1;
      continue;
    }
    const marker = buf[i + 1];
    // SOF0..SOF15 except DHT(c4), JPGA(c8), DAC(cc)
    if (marker >= 0xc0 && marker <= 0xcf && marker !== 0xc4 && marker !== 0xc8 && marker !== 0xcc) {
      return { height: buf.readUInt16BE(i + 5), width: buf.readUInt16BE(i + 7) };
    }
    const len = buf.readUInt16BE(i + 2);
    if (len < 2) return null;
    i += 2 + len;
  }
  return null;
}

function parseStreamInfo(stderr) {
  const info = {};
  const res = stderr.match(/Video:.*?,\s*(\d{2,5})x(\d{2,5})/);
  if (res) {
    info.width = Number(res[1]);
    info.height = Number(res[2]);
  }
  const fps = stderr.match(/,\s*([\d.]+)\s*fps/);
  if (fps) info.fps = Number(fps[1]);
  const codec = stderr.match(/Video:\s*([a-z0-9]+)/i);
  if (codec) info.codec = codec[1];
  return info;
}

class Source {
  constructor(key, rtspUrl, fps) {
    this.key = key;
    this.rtspUrl = rtspUrl;
    this.fps = fps;
    this.ffmpeg = null;
    this.buffer = Buffer.alloc(0);
    this.latest = null;
    this.latestAt = 0;
    this.frames = 0;
    this.startedAt = Date.now();
    this.lastReadAt = Date.now();
    this.error = null;
    this.stderr = '';
    this.camera = {};
    this.measured = { count: 0, since: Date.now(), fps: 0 };
  }

  spawn() {
    const args = [
      '-hide_banner',
      '-loglevel',
      'info',
      '-rtsp_transport',
      'tcp',
      '-i',
      this.rtspUrl,
      '-an',
      '-r',
      String(this.fps),
      '-f',
      'image2pipe',
      '-vcodec',
      'mjpeg',
      '-q:v',
      '4',
      '-',
    ];

    const ff = spawn('ffmpeg', args, { windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'] });
    this.ffmpeg = ff;
    this.error = null;

    ff.stdout.on('data', (chunk) => this.consume(chunk));

    ff.stderr.on('data', (d) => {
      this.stderr += d.toString();
      if (this.stderr.length > 8000) this.stderr = this.stderr.slice(-4000);
      if (!this.camera.width) {
        const info = parseStreamInfo(this.stderr);
        if (info.width) this.camera = info;
      }
    });

    ff.on('close', (code) => {
      if (this.ffmpeg !== ff) return;
      this.ffmpeg = null;
      if (code !== 0 && code !== null) {
        this.error = this.stderr.trim().split('\n').slice(-2).join(' ') || `ffmpeg exited ${code}`;
      }
    });
  }

  consume(chunk) {
    this.buffer = this.buffer.length ? Buffer.concat([this.buffer, chunk]) : chunk;
    if (this.buffer.length > MAX_BUFFER) this.buffer = this.buffer.subarray(-MAX_BUFFER);

    for (;;) {
      const start = this.buffer.indexOf(SOI);
      if (start < 0) return;
      const end = this.buffer.indexOf(EOI, start + 3);
      if (end < 0) {
        if (start > 0) this.buffer = this.buffer.subarray(start);
        return;
      }
      this.latest = this.buffer.subarray(start, end + 2);
      this.latestAt = Date.now();
      this.frames += 1;
      this.measured.count += 1;
      this.buffer = this.buffer.subarray(end + 2);

      const elapsed = this.latestAt - this.measured.since;
      if (elapsed >= 2000) {
        this.measured.fps = Number(((this.measured.count * 1000) / elapsed).toFixed(2));
        this.measured.count = 0;
        this.measured.since = this.latestAt;
      }
      if (!this.camera.width) {
        const size = jpegSize(this.latest);
        if (size) this.camera = { ...this.camera, ...size };
      }
    }
  }

  stop() {
    const ff = this.ffmpeg;
    this.ffmpeg = null;
    if (ff && !ff.killed) {
      ff.kill('SIGTERM');
      setTimeout(() => {
        if (!ff.killed) ff.kill('SIGKILL');
      }, 1500);
    }
  }
}

export class FrameSource {
  constructor({ fps = 4 } = {}) {
    this.fps = fps;
    this.sources = new Map();
    this.sweeper = setInterval(() => this.sweep(), 15_000);
    if (this.sweeper.unref) this.sweeper.unref();
  }

  ensure(cameraId, rtspUrl) {
    let src = this.sources.get(cameraId);
    if (src && src.rtspUrl !== rtspUrl) {
      src.stop();
      this.sources.delete(cameraId);
      src = null;
    }
    if (!src) {
      src = new Source(cameraId, rtspUrl, this.fps);
      this.sources.set(cameraId, src);
      src.spawn();
    } else if (!src.ffmpeg && (!src.latestAt || Date.now() - src.latestAt > 5000)) {
      // ffmpeg died and the cached frame went stale — reconnect
      src.error = null;
      src.spawn();
    }
    src.lastReadAt = Date.now();
    return src;
  }

  /** Latest decoded JPEG, waiting up to timeoutMs for the first frame. */
  async grab(cameraId, rtspUrl, { timeoutMs = 12_000 } = {}) {
    const src = this.ensure(cameraId, rtspUrl);
    const startedWaiting = Date.now();
    const seenAt = src.latestAt;

    while (Date.now() - startedWaiting < timeoutMs) {
      if (src.latest && src.latestAt !== 0 && (src.latestAt > seenAt || Date.now() - src.latestAt < 2000)) {
        return this.describe(src);
      }
      if (src.error && !src.latest) throw new Error(src.error);
      if (!src.ffmpeg && !src.latest) {
        src.spawn();
      }
      await new Promise((r) => setTimeout(r, 80));
    }

    if (src.latest) return this.describe(src);
    throw new Error(src.error || 'No frame from camera (RTSP timeout)');
  }

  describe(src) {
    const size = src.camera.width ? src.camera : jpegSize(src.latest) || {};
    return {
      jpeg: src.latest,
      capturedAt: src.latestAt,
      ageMs: Date.now() - src.latestAt,
      frames: src.frames,
      sourceFps: src.camera.fps ?? null,
      captureFps: src.measured.fps || null,
      width: size.width ?? null,
      height: size.height ?? null,
      codec: src.camera.codec ?? null,
      error: src.error,
    };
  }

  stats(cameraId) {
    const src = this.sources.get(cameraId);
    if (!src) return null;
    return this.describe(src);
  }

  release(cameraId) {
    const src = this.sources.get(cameraId);
    if (!src) return false;
    src.stop();
    this.sources.delete(cameraId);
    return true;
  }

  sweep() {
    const now = Date.now();
    for (const [key, src] of this.sources) {
      if (now - src.lastReadAt > IDLE_STOP_MS) {
        src.stop();
        this.sources.delete(key);
      }
    }
  }

  stopAll() {
    for (const key of [...this.sources.keys()]) this.release(key);
    clearInterval(this.sweeper);
  }
}
