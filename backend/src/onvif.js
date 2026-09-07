import onvif from 'onvif';
import { getLocalSubnets } from './network.js';

const { Cam, Discovery } = onvif;

function camGetDeviceInformation(cam) {
  return new Promise((resolve, reject) => {
    cam.getDeviceInformation((err, info) => {
      if (err) reject(err);
      else resolve(info);
    });
  });
}

function camGetStreamUri(cam, profileToken) {
  return new Promise((resolve, reject) => {
    cam.getStreamUri({ protocol: 'RTSP', profileToken }, (err, stream) => {
      if (err) reject(err);
      else resolve(stream);
    });
  });
}

function connectCam(options) {
  return new Promise((resolve, reject) => {
    const cam = new Cam(options, (err) => {
      if (err) reject(err);
      else resolve(cam);
    });
  });
}

export function discoverOnvif(timeoutMs = 4000) {
  return new Promise((resolve) => {
    const devices = [];
    let settled = false;

    const finish = () => {
      if (settled) return;
      settled = true;
      resolve(devices);
    };

    try {
      Discovery.probe({ timeout: timeoutMs }, (err, cams) => {
        if (!err && Array.isArray(cams)) {
          for (const cam of cams) {
            devices.push({
              ip: cam.hostname || cam.xaddrs?.[0]?.match(/\/\/([^/:]+)/)?.[1],
              hostname: cam.hostname,
              port: cam.port || 80,
              path: cam.path,
              xaddrs: cam.xaddrs || [],
              name: cam.name || cam.hostname || 'ONVIF Camera',
              source: 'onvif',
            });
          }
        }
        finish();
      });
    } catch {
      finish();
    }

    setTimeout(finish, timeoutMs + 500);
  });
}

export async function connectOnvifCamera({ ip, port = 80, username, password, path }) {
  const cam = await connectCam({
    hostname: ip,
    username: username || 'admin',
    password: password || '',
    port: Number(port) || 80,
    path: path || '/onvif/device_service',
  });

  let info = {};
  try {
    info = await camGetDeviceInformation(cam);
  } catch {
    /* some cams skip this */
  }

  const profiles = cam.profiles || [];
  const streams = [];

  for (const profile of profiles) {
    try {
      const stream = await camGetStreamUri(cam, profile.$.token);
      if (stream?.uri) {
        streams.push({
          profile: profile.name || profile.$.token,
          token: profile.$.token,
          uri: stream.uri,
        });
      }
    } catch {
      /* skip broken profile */
    }
  }

  if (!streams.length && profiles[0]) {
    try {
      const stream = await camGetStreamUri(cam, profiles[0].$.token);
      if (stream?.uri) {
        streams.push({
          profile: profiles[0].name || 'main',
          token: profiles[0].$.token,
          uri: stream.uri,
        });
      }
    } catch {
      /* ignore */
    }
  }

  return {
    ip,
    port,
    manufacturer: info.manufacturer || null,
    model: info.model || null,
    firmware: info.firmwareVersion || null,
    serial: info.serialNumber || null,
    profiles: profiles.map((p) => ({
      name: p.name,
      token: p.$?.token,
    })),
    streams,
    rtspUrl: streams[0]?.uri || null,
  };
}

export function guessLocalHints() {
  return getLocalSubnets().map((s) => ({
    interface: s.name,
    ip: s.address,
    subnet: `${s.prefix}.0/24`,
  }));
}
