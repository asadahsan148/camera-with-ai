# WiFi Cameras — React + Node

Discover IP cameras on your local WiFi, connect over ONVIF/RTSP, and live-stream in the browser.

## Requirements

- Node.js 18+
- ffmpeg on PATH (already available if `ffmpeg -version` works)

## Setup

```bash
npm run install:all
```

## Run

Terminal 1 — backend:

```bash
npm run dev:backend
```

Terminal 2 — frontend:

```bash
npm run dev:frontend
```

Open http://localhost:5173

## Usage

1. Enter camera username/password (often `admin`).
2. Click **Scan WiFi cameras** — ONVIF + SSDP + subnet port scan.
3. Click **Connect**, then **Live**.
4. If the stream is blank, edit the RTSP URL in the details panel.

### Common RTSP paths

- Hikvision: `rtsp://USER:PASS@IP:554/Streaming/Channels/101`
- Dahua: `rtsp://USER:PASS@IP:554/cam/realmonitor?channel=1&subtype=0`
- Many WiFi cams: `rtsp://USER:PASS@IP:554/stream1`

## Snooker AI monitor (testing)

Separate Python worker analyzes JPEG frames from RTSP and emits:

- `RACK_DETECTED`
- `FRAME_STARTED`
- `FRAME_ENDED`

```bash
pip install -r ai-worker/requirements.txt
npm run dev:ai
```

Then in the UI: connect a table camera → **AI Monitor**. Events appear in the bottom panel and backend console.
