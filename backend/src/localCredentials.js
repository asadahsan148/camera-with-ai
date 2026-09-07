/**
 * Local NVR/camera login — stored on disk so UI does not ask every restart.
 * File is gitignored; do not commit real passwords.
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FILE = path.join(HERE, '..', 'data', 'local-credentials.json');

const DEFAULTS = {
  username: 'admin',
  password: 'Raja7860',
};

export function loadLocalCredentials() {
  try {
    const raw = JSON.parse(fs.readFileSync(FILE, 'utf8'));
    return {
      username: String(raw.username ?? DEFAULTS.username),
      password: String(raw.password ?? DEFAULTS.password),
    };
  } catch {
    return { ...DEFAULTS };
  }
}

export function saveLocalCredentials({ username, password }) {
  const next = {
    username: username != null && username !== '' ? String(username) : DEFAULTS.username,
    password: password != null ? String(password) : '',
    updated_at: new Date().toISOString(),
  };
  fs.mkdirSync(path.dirname(FILE), { recursive: true });
  const tmp = `${FILE}.${process.pid}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(next, null, 2), 'utf8');
  fs.renameSync(tmp, FILE);
  return { username: next.username, password: next.password };
}

/** Ensure the file exists with the club default on first boot. */
export function ensureLocalCredentials() {
  if (!fs.existsSync(FILE)) {
    return saveLocalCredentials(DEFAULTS);
  }
  return loadLocalCredentials();
}
