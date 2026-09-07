import os from 'os';
import net from 'net';
import dgram from 'dgram';

export function getLocalSubnets() {
  const interfaces = os.networkInterfaces();
  const subnets = [];

  for (const [name, addrs] of Object.entries(interfaces)) {
    if (!addrs) continue;
    for (const addr of addrs) {
      if (addr.family !== 'IPv4' || addr.internal) continue;
      if (addr.address.startsWith('169.254.')) continue;
      const parts = addr.address.split('.').map(Number);
      const maskParts = (addr.netmask || '255.255.255.0').split('.').map(Number);
      const network = parts.map((p, i) => p & maskParts[i]);
      const broadcast = parts.map((p, i) => p | (~maskParts[i] & 255));
      subnets.push({
        name,
        address: addr.address,
        netmask: addr.netmask,
        network: network.join('.'),
        broadcast: broadcast.join('.'),
        prefix: `${parts[0]}.${parts[1]}.${parts[2]}`,
      });
    }
  }

  return subnets;
}

function probePort(host, port, timeoutMs = 400) {
  return new Promise((resolve) => {
    const socket = new net.Socket();
    let done = false;

    const finish = (open) => {
      if (done) return;
      done = true;
      socket.destroy();
      resolve(open);
    };

    socket.setTimeout(timeoutMs);
    socket.once('connect', () => finish(true));
    socket.once('timeout', () => finish(false));
    socket.once('error', () => finish(false));
    socket.connect(port, host);
  });
}

async function mapPool(items, concurrency, worker) {
  const results = [];
  let index = 0;

  async function run() {
    while (index < items.length) {
      const current = index++;
      results[current] = await worker(items[current], current);
    }
  }

  await Promise.all(Array.from({ length: Math.min(concurrency, items.length) }, run));
  return results;
}

function hostsInSubnet(prefix) {
  const hosts = [];
  for (let i = 1; i <= 254; i++) hosts.push(`${prefix}.${i}`);
  return hosts;
}

const CAMERA_PORTS = [
  { port: 554, protocol: 'rtsp' },
  { port: 8554, protocol: 'rtsp' },
  { port: 10554, protocol: 'rtsp' },
  { port: 80, protocol: 'http' },
  { port: 8080, protocol: 'http' },
  { port: 8000, protocol: 'http' },
  { port: 37777, protocol: 'dahua' },
  { port: 34567, protocol: 'xmeye' },
];

export async function scanNetworkForCameras(onProgress) {
  const subnets = getLocalSubnets();
  if (!subnets.length) {
    return { subnets: [], cameras: [], stats: { hostsChecked: 0, openHosts: 0 } };
  }

  const found = new Map();
  const hosts = [...new Set(subnets.flatMap((s) => hostsInSubnet(s.prefix)))];
  const selfIps = new Set(subnets.map((s) => s.address));
  let checked = 0;
  let openHosts = 0;

  await mapPool(hosts, 64, async (host) => {
    const openPorts = [];
    for (const entry of CAMERA_PORTS) {
      if (await probePort(host, entry.port)) {
        openPorts.push(entry);
      }
    }

    checked += 1;
    if (onProgress && checked % 20 === 0) {
      onProgress({ checked, total: hosts.length });
    }

    if (!openPorts.length) return;
    if (selfIps.has(host)) return;
    openHosts += 1;

    const hasRtsp = openPorts.some((p) => p.protocol === 'rtsp');
    const hasCamVendor = openPorts.some((p) => p.protocol === 'dahua' || p.protocol === 'xmeye');
    // Only real camera/NVR signals — skip plain HTTP routers/printers
    if (!hasRtsp && !hasCamVendor) return;

    const primary = openPorts.find((p) => p.protocol === 'rtsp') || openPorts[0];
    const brandHint = openPorts.some((p) => p.protocol === 'dahua')
      ? 'dahua'
      : openPorts.some((p) => p.protocol === 'xmeye')
        ? 'xmeye'
        : null;

    found.set(host, {
      ip: host,
      ports: openPorts,
      protocol: primary.protocol,
      brandHint,
      rtspUrl: null,
      suggestedRtsp: hasRtsp
        ? brandHint === 'dahua'
          ? `rtsp://${host}:${primary.port}/cam/realmonitor?channel=1&subtype=0`
          : `rtsp://${host}:${primary.port}/`
        : brandHint === 'dahua'
          ? `rtsp://${host}:554/cam/realmonitor?channel=1&subtype=0`
          : null,
      source: 'network-scan',
    });
  });

  return {
    subnets,
    cameras: [...found.values()],
    stats: {
      hostsChecked: checked,
      openHosts,
      included: found.size,
    },
  };
}

export function ssdpDiscover(timeoutMs = 2500) {
  return new Promise((resolve) => {
    const socket = dgram.createSocket({ type: 'udp4', reuseAddr: true });
    const devices = new Map();
    const MSEARCH =
      'M-SEARCH * HTTP/1.1\r\n' +
      'HOST: 239.255.255.250:1900\r\n' +
      'MAN: "ssdp:discover"\r\n' +
      'MX: 2\r\n' +
      'ST: ssdp:all\r\n\r\n';

    const finish = () => {
      try {
        socket.close();
      } catch {
        /* ignore */
      }
      resolve([...devices.values()]);
    };

    socket.on('message', (msg, rinfo) => {
      const text = msg.toString();
      const location = /LOCATION:\s*(.+)/i.exec(text)?.[1]?.trim();
      const server = /SERVER:\s*(.+)/i.exec(text)?.[1]?.trim();
      const st = /ST:\s*(.+)/i.exec(text)?.[1]?.trim();
      const usn = /USN:\s*(.+)/i.exec(text)?.[1]?.trim();
      const looksLikeCamera =
        /camera|onvif|ipcam|nvr|dvr|rtsp|hikvision|dahua|axis|reolink/i.test(
          `${server || ''} ${st || ''} ${usn || ''} ${location || ''}`
        );

      if (!looksLikeCamera && !/onvif/i.test(text)) return;

      devices.set(rinfo.address, {
        ip: rinfo.address,
        location: location || null,
        server: server || null,
        st: st || null,
        source: 'ssdp',
      });
    });

    socket.on('error', () => finish());

    socket.bind(() => {
      try {
        socket.setBroadcast(true);
        socket.send(Buffer.from(MSEARCH), 1900, '239.255.255.250');
      } catch {
        finish();
        return;
      }
      setTimeout(finish, timeoutMs);
    });
  });
}
