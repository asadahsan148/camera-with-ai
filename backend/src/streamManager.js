import { spawn } from 'child_process';
import { WebSocketServer } from 'ws';
import { randomUUID } from 'crypto';

export class StreamManager {
  constructor() {
    this.streams = new Map();
  }

  list() {
    return [...this.streams.values()].map((s) => ({
      id: s.id,
      cameraId: s.cameraId,
      name: s.name,
      rtspUrl: this.maskUrl(s.rtspUrl),
      clients: s.clients.size,
      running: Boolean(s.ffmpeg) && !s.ffmpeg.killed,
      startedAt: s.startedAt,
      error: s.error,
    }));
  }

  maskUrl(url) {
    try {
      const u = new URL(url);
      if (u.password) u.password = '****';
      return u.toString();
    } catch {
      return url.replace(/\/\/([^:]+):([^@]+)@/, '//$1:****@');
    }
  }

  start({ cameraId, name, rtspUrl }) {
    const existing = [...this.streams.values()].find(
      (s) => s.rtspUrl === rtspUrl && s.cameraId === cameraId
    );
    if (existing && existing.ffmpeg && !existing.ffmpeg.killed) {
      return existing;
    }

    if (existing) {
      this.stop(existing.id);
    }

    const id = randomUUID();
    const stream = {
      id,
      cameraId,
      name: name || cameraId || 'Camera',
      rtspUrl,
      clients: new Set(),
      ffmpeg: null,
      startedAt: new Date().toISOString(),
      error: null,
      wss: null,
    };

    this.streams.set(id, stream);
    this.spawnFfmpeg(stream);
    return stream;
  }

  attachServer(server) {
    this.wss = new WebSocketServer({ noServer: true });

    server.on('upgrade', (req, socket, head) => {
      const match = req.url?.match(/^\/stream\/([^/?]+)/);
      if (!match) {
        socket.destroy();
        return;
      }

      const streamId = match[1];
      const stream = this.streams.get(streamId);
      if (!stream) {
        socket.destroy();
        return;
      }

      this.wss.handleUpgrade(req, socket, head, (ws) => {
        stream.clients.add(ws);
        ws.on('close', () => {
          stream.clients.delete(ws);
          if (stream.clients.size === 0) {
            setTimeout(() => {
              if (stream.clients.size === 0) this.stop(streamId);
            }, 15000);
          }
        });
      });
    });
  }

  spawnFfmpeg(stream) {
    const args = [
      '-rtsp_transport',
      'tcp',
      '-i',
      stream.rtspUrl,
      '-f',
      'mpegts',
      '-codec:v',
      'mpeg1video',
      '-r',
      '25',
      '-b:v',
      '1200k',
      '-bf',
      '0',
      '-an',
      '-',
    ];

    const ffmpeg = spawn('ffmpeg', args, {
      windowsHide: true,
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    stream.ffmpeg = ffmpeg;
    stream.error = null;

    ffmpeg.stdout.on('data', (chunk) => {
      for (const client of stream.clients) {
        if (client.readyState === 1) {
          client.send(chunk);
        }
      }
    });

    let errBuf = '';
    ffmpeg.stderr.on('data', (data) => {
      errBuf += data.toString();
      if (errBuf.length > 4000) errBuf = errBuf.slice(-2000);
    });

    ffmpeg.on('close', (code) => {
      if (stream.ffmpeg === ffmpeg) {
        stream.ffmpeg = null;
        if (code !== 0 && code !== null) {
          stream.error = errBuf.trim().split('\n').slice(-3).join(' ') || `ffmpeg exited ${code}`;
        }
        for (const client of stream.clients) {
          try {
            client.close();
          } catch {
            /* ignore */
          }
        }
        stream.clients.clear();
      }
    });
  }

  stop(id) {
    const stream = this.streams.get(id);
    if (!stream) return false;

    if (stream.ffmpeg && !stream.ffmpeg.killed) {
      stream.ffmpeg.kill('SIGTERM');
      setTimeout(() => {
        if (stream.ffmpeg && !stream.ffmpeg.killed) stream.ffmpeg.kill('SIGKILL');
      }, 2000);
    }

    for (const client of stream.clients) {
      try {
        client.close();
      } catch {
        /* ignore */
      }
    }

    this.streams.delete(id);
    return true;
  }

  stopAll() {
    for (const id of [...this.streams.keys()]) {
      this.stop(id);
    }
  }
}
