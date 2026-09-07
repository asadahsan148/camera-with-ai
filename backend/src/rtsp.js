import { spawn } from 'child_process';

const DAHUA_PATH = '/cam/realmonitor?channel=1&subtype=0';
const HIKVISION_PATH = '/Streaming/Channels/101';
const GENERIC_PATHS = ['/stream1', '/Streaming/Channels/1', '/h264', '/live/ch00_0', '/'];

export function buildRtspCandidates({ ip, ports = [], username, password, brandHint }) {
  const auth =
    username != null
      ? `${encodeURIComponent(username)}:${encodeURIComponent(password || '')}@`
      : '';
  const rtspPort = ports.find((p) => p.protocol === 'rtsp')?.port || 554;
  const isDahua = brandHint === 'dahua' || ports.some((p) => p.protocol === 'dahua');
  const isXmeye = ports.some((p) => p.protocol === 'xmeye');

  const paths = [];
  if (isDahua) paths.push(DAHUA_PATH, HIKVISION_PATH, ...GENERIC_PATHS);
  else if (isXmeye) paths.push('/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp', ...GENERIC_PATHS);
  else paths.push(HIKVISION_PATH, DAHUA_PATH, ...GENERIC_PATHS);

  return [...new Set(paths)].map((path) => `rtsp://${auth}${ip}:${rtspPort}${path}`);
}

export function probeRtsp(url, timeoutMs = 8000) {
  return new Promise((resolve) => {
    const args = [
      '-hide_banner',
      '-rtsp_transport',
      'tcp',
      '-i',
      url,
      '-t',
      '1',
      '-f',
      'null',
      '-',
    ];

    const ff = spawn('ffmpeg', args, {
      windowsHide: true,
      stdio: ['ignore', 'ignore', 'pipe'],
    });

    let err = '';
    let settled = false;

    const finish = (result) => {
      if (settled) return;
      settled = true;
      try {
        if (!ff.killed) ff.kill('SIGKILL');
      } catch {
        /* ignore */
      }
      resolve(result);
    };

    const timer = setTimeout(() => {
      finish({ ok: false, error: 'Timeout while probing RTSP', url });
    }, timeoutMs);

    ff.stderr.on('data', (chunk) => {
      err += chunk.toString();
    });

    ff.on('close', (code) => {
      clearTimeout(timer);
      const text = err.toLowerCase();
      if (text.includes('401') || text.includes('unauthorized') || text.includes('authorization failed')) {
        finish({ ok: false, error: 'Wrong username/password (401 Unauthorized)', code: 401, url });
        return;
      }
      if (text.includes('404') || text.includes('not found')) {
        finish({ ok: false, error: 'RTSP path not found (404). Try another stream path.', code: 404, url });
        return;
      }
      if (text.includes('connection refused') || text.includes('timed out') || text.includes('connection timed out')) {
        finish({ ok: false, error: 'Cannot reach camera RTSP port', code: 503, url });
        return;
      }
      if (code === 0 || /stream\s*#0|video:|input #0/i.test(err)) {
        finish({ ok: true, url, detail: 'RTSP OK' });
        return;
      }
      const last = err.trim().split(/\r?\n/).slice(-2).join(' ');
      finish({ ok: false, error: last || `ffmpeg exited ${code}`, url });
    });
  });
}

export async function findWorkingRtsp({ ip, ports, username, password, brandHint, preferredUrl }) {
  const candidates = [];
  if (preferredUrl) candidates.push(preferredUrl);
  candidates.push(...buildRtspCandidates({ ip, ports, username, password, brandHint }));

  const unique = [...new Set(candidates)];
  let lastAuthError = null;

  for (const url of unique) {
    const result = await probeRtsp(url);
    if (result.ok) return result;
    if (result.code === 401) lastAuthError = result;
    // if auth failed on first path, still try other paths in case path was wrong with open auth,
    // but usually 401 means credentials are wrong for whole camera
    if (result.code === 401 && username) {
      // keep trying a couple paths; some firmwares 401 on wrong path oddly
      continue;
    }
  }

  return (
    lastAuthError || {
      ok: false,
      error: 'No working RTSP URL found. Check IP, path, and camera login.',
    }
  );
}
