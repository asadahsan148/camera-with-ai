import { createHash, randomBytes } from 'crypto';

function md5(text) {
  return createHash('md5').update(text, 'utf8').digest('hex');
}

function parseDigestChallenge(header) {
  if (!header || !/digest/i.test(header)) return null;
  const params = {};
  const re = /(\w+)=(?:"([^"]*)"|([^,\s]*))/g;
  let match;
  while ((match = re.exec(header))) {
    params[match[1].toLowerCase()] = match[2] ?? match[3] ?? '';
  }
  return params;
}

function buildDigestHeader({ username, password, method, uri, challenge, nc = '00000001' }) {
  const realm = challenge.realm || '';
  const nonce = challenge.nonce || '';
  const qop = (challenge.qop || '').split(',')[0].trim();
  const opaque = challenge.opaque;
  const algorithm = (challenge.algorithm || 'MD5').toUpperCase();
  const cnonce = randomBytes(8).toString('hex');

  const ha1 =
    algorithm === 'MD5-SESS'
      ? md5(`${md5(`${username}:${realm}:${password}`)}:${nonce}:${cnonce}`)
      : md5(`${username}:${realm}:${password}`);
  const ha2 = md5(`${method}:${uri}`);

  const response = qop
    ? md5(`${ha1}:${nonce}:${nc}:${cnonce}:${qop}:${ha2}`)
    : md5(`${ha1}:${nonce}:${ha2}`);

  let header =
    `Digest username="${username}", realm="${realm}", nonce="${nonce}", uri="${uri}", ` +
    `algorithm=${algorithm || 'MD5'}, response="${response}"`;

  if (qop) header += `, qop=${qop}, nc=${nc}, cnonce="${cnonce}"`;
  if (opaque !== undefined && opaque !== null && opaque !== '') {
    header += `, opaque="${opaque}"`;
  }

  return header;
}

function basicAuthHeader(username, password) {
  return `Basic ${Buffer.from(`${username}:${password || ''}`).toString('base64')}`;
}

async function rawGet(url, headers, timeoutMs) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, { headers, signal: controller.signal });
    const text = await res.text();
    return { res, text };
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Dahua CGI with Digest (primary) + Basic fallback.
 */
export async function dahuaGet(ip, path, username, password, timeoutMs = 10000) {
  const url = `http://${ip}${path}`;
  const user = username || 'admin';
  const pass = password ?? '';

  // 1) Probe for challenge (no auth)
  const first = await rawGet(url, {}, timeoutMs);
  if (first.res.ok) return first.text;

  const www = first.res.headers.get('www-authenticate') || '';
  const challenge = parseDigestChallenge(www);

  // 2) Digest auth
  if (challenge) {
    const auth = buildDigestHeader({
      username: user,
      password: pass,
      method: 'GET',
      uri: path,
      challenge,
    });
    const second = await rawGet(url, { Authorization: auth }, timeoutMs);
    if (second.res.ok) return second.text;

    // Some firmwares want URI without query for HA2
    if (path.includes('?')) {
      const pathOnly = path.split('?')[0];
      const auth2 = buildDigestHeader({
        username: user,
        password: pass,
        method: 'GET',
        uri: pathOnly,
        challenge: parseDigestChallenge(second.res.headers.get('www-authenticate') || www),
        nc: '00000002',
      });
      const third = await rawGet(url, { Authorization: auth2 }, timeoutMs);
      if (third.res.ok) return third.text;

      if (third.res.status === 401) {
        const err = new Error(
          'NVR ne login reject kiya (Digest 401). Username/password dobara check karein — yeh Dahua web panel wala hona chahiye.'
        );
        err.status = 401;
        err.body = third.text;
        throw err;
      }

      const err = new Error(`Dahua API HTTP ${third.res.status}`);
      err.status = third.res.status;
      throw err;
    }

    if (second.res.status === 401) {
      const err = new Error(
        'NVR ne login reject kiya (Digest 401). Username/password dobara check karein — yeh Dahua web panel wala hona chahiye.'
      );
      err.status = 401;
      throw err;
    }

    const err = new Error(`Dahua API HTTP ${second.res.status}`);
    err.status = second.res.status;
    throw err;
  }

  // 3) Basic fallback
  const basic = await rawGet(
    url,
    { Authorization: basicAuthHeader(user, pass) },
    timeoutMs
  );
  if (basic.res.ok) return basic.text;

  const err = new Error(
    basic.res.status === 401
      ? 'Wrong username/password for Dahua NVR'
      : `Dahua API HTTP ${basic.res.status}`
  );
  err.status = basic.res.status;
  throw err;
}

function parseKeyValueTable(text) {
  const rows = {};
  for (const line of String(text || '').split(/\r?\n/)) {
    const idx = line.indexOf('=');
    if (idx === -1) continue;
    rows[line.slice(0, idx).trim()] = line.slice(idx + 1).trim();
  }
  return rows;
}

export async function fetchDahuaChannels({ ip, username, password, maxChannels = 32 }) {
  const channels = [];
  let deviceName = null;
  let deviceType = null;

  // Auth check first — fail fast with clear error
  const typeText = await dahuaGet(
    ip,
    '/cgi-bin/magicBox.cgi?action=getDeviceType',
    username,
    password
  );
  deviceType = parseKeyValueTable(typeText).type || typeText.trim();

  try {
    const nameText = await dahuaGet(
      ip,
      '/cgi-bin/magicBox.cgi?action=getMachineName',
      username,
      password
    );
    deviceName = parseKeyValueTable(nameText).name || null;
  } catch {
    /* optional */
  }

  let titles = {};
  try {
    const titleText = await dahuaGet(
      ip,
      '/cgi-bin/configManager.cgi?action=getConfig&name=ChannelTitle',
      username,
      password
    );
    const table = parseKeyValueTable(titleText);
    for (const [key, value] of Object.entries(table)) {
      const m = /ChannelTitle\[(\d+)\]\.Name/i.exec(key);
      if (m) titles[Number(m[1])] = value;
    }
  } catch {
    /* optional */
  }

  try {
    const camText = await dahuaGet(
      ip,
      '/cgi-bin/LogicDeviceManager.cgi?action=getCameraAll',
      username,
      password
    );
    const table = parseKeyValueTable(camText);
    const byUnique = new Map();
    for (const [key, value] of Object.entries(table)) {
      const m = /cameras\[(\d+)\]\.(.+)$/i.exec(key);
      if (!m) continue;
      const idx = Number(m[1]);
      const field = m[2];
      if (!byUnique.has(idx)) byUnique.set(idx, {});
      byUnique.get(idx)[field] = value;
    }
    for (const [, cam] of byUnique) {
      const channel = Number(cam.UniqueChannel ?? cam.Channel ?? cam.channel);
      if (!Number.isFinite(channel)) continue;
      const oneBased = channel >= 0 ? channel + 1 : channel;
      const name = cam.Name || cam.name || titles[channel] || titles[oneBased - 1];
      if (name) titles[oneBased - 1] = name;
    }
  } catch {
    /* optional */
  }

  // Also try encode channels / max remote
  if (!Object.keys(titles).length) {
    try {
      const enc = await dahuaGet(
        ip,
        '/cgi-bin/configManager.cgi?action=getConfig&name=Encode',
        username,
        password
      );
      const table = parseKeyValueTable(enc);
      const idxs = new Set();
      for (const key of Object.keys(table)) {
        const m = /Encode\[(\d+)\]/i.exec(key);
        if (m) idxs.add(Number(m[1]));
      }
      for (const idx of [...idxs].sort((a, b) => a - b)) {
        titles[idx] = `Channel ${idx + 1}`;
      }
    } catch {
      /* optional */
    }
  }

  const indexes = Object.keys(titles)
    .map(Number)
    .filter((n) => Number.isFinite(n))
    .sort((a, b) => a - b);

  if (indexes.length) {
    for (const idx of indexes) {
      const channel = idx + 1;
      const name = String(titles[idx] || `Channel ${channel}`).trim();
      if (!name || /^null$/i.test(name)) continue;
      channels.push({
        channel,
        name,
        rtspPath: `/cam/realmonitor?channel=${channel}&subtype=0`,
      });
    }
  } else {
    for (let channel = 1; channel <= Math.min(16, maxChannels); channel++) {
      channels.push({
        channel,
        name: `Channel ${channel}`,
        rtspPath: `/cam/realmonitor?channel=${channel}&subtype=0`,
      });
    }
  }

  return {
    ip,
    deviceName,
    deviceType,
    channels: channels.slice(0, maxChannels),
  };
}

export function buildDahuaRtsp({ ip, channel, username, password, subtype = 0 }) {
  const auth = `${encodeURIComponent(username)}:${encodeURIComponent(password || '')}@`;
  return `rtsp://${auth}${ip}:554/cam/realmonitor?channel=${channel}&subtype=${subtype}`;
}
