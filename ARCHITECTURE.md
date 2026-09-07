# SCOS — Architecture & Project Memory

**SCOS** (Snooker Club Operating System) is a local CCTV-based snooker vision stack.  
This document is the living memory of **what we are building**, **how the pieces fit**, and **where we are now**.

---

## 1. What we are trying to achieve

Build a **club-ready snooker vision product** that:

1. Connects to real Dahua/NVR CCTV (multi-channel RTSP).
2. Calibrates each physical table once (perspective-correct top-down view).
3. Detects balls reliably on that warped table view.
4. Later: tracking → scoring → shot / frame-end events → club SaaS features.

**Hard product constraints (current decisions):**

| Constraint | Decision |
|------------|----------|
| Recurring cost | Prefer **local inference**, no per-frame cloud API in production |
| Internet | Production detector must not depend on continuous cloud inference |
| Licensing | Production model must be **commercially usable for SaaS** (prefer Apache 2.0 / MIT) |
| Manual labeling | Annotation tool is **optional / debug** — not the primary path to hundreds of labels |
| Stability | Do **not** break camera streaming, calibration persistence, or perspective warp |

**Long-term vision pipeline (target):**

```
Camera (RTSP)
  → Phase 1 table calibration (4 corners + warp)
  → Perspective-corrected table frame
  → Local ball detector (production)
  → Tracking
  → Scoring / shot / frame-end
```

---

## 2. Current stage (as of 2026-09)

### Status summary

| Area | Status | Notes |
|------|--------|--------|
| Camera / NVR connect | **Done** | Dahua channels, RTSP, browser live stream |
| Credentials persistence | **Done** | `backend/data/local-credentials.json` (gitignored) |
| **Phase 1 — table calibration** | **Done** | Geometry only; persists across restarts |
| Phase 2 — dataset capture / annotate | **Done (optional tool)** | Not the primary production path |
| Phase 2B — pretrained Roboflow benchmark | **Scaffolded (diagnostic only)** | Hosted API adapter exists; **not approved for production** |
| Production local detector | **Not started** | Research concluded: avoid Roboflow Universe YOLO for SaaS |
| Tracking / scoring / shot / frame-end | **Not started** | Explicitly deferred |

### AI / calibration stage (detail)

**Phase 1 (table geometry) is the current solid baseline.**

- Admin clicks **four corners** (TL → TR → BR → BL) on a live CCTV frame.
- AI worker computes a **perspective matrix** and warps to a consistent 2:1 top-down rectangle.
- A **margin** (~10% of surface height) keeps cushions / pocket jaws in view.
- Calibration is stored in `backend/data/tables.json` with:
  - `corners` + `corners_normalized`
  - `perspective_matrix`
  - `surface_width`, `aspect`, `margin`, output size
  - `camera_key` = `nvrIp|chN` so calibration survives camera UUID churn after “Load all channels”
- UI: `/table-calibration`
- This step is **pure geometry** — no ball detection, no colour logic, no scoring.

**What “AI calibration” means today:**  
It means **table/camera geometry calibration**, not model training. Ball AI is a separate later stage that consumes the warped frame.

**What is next for ball AI (agreed direction, not yet implemented):**

1. Do **not** productionize `snooker-ball-detection-rnhxo-95km5/2` (Roboflow hosted / AGPL-family / paid commercial self-host).
2. Prefer training a **local** detector with a permissive license, e.g. **RF-DETR (Apache 2.0)** or **RTMDet (MIT)**, export **ONNX**, run in the AI worker.
3. Use Phase 1 warped frames (and optional dataset tool / license-clean public datasets) for training data when ready.

---

## 3. Runtime topology

Three processes (local Windows / Laragon):

| Service | Port | Role |
|---------|------|------|
| Frontend (Vite + React) | `5173` | UI |
| Backend (Express) | `5050` | Cameras, streams, tables, dataset, model-test APIs |
| AI worker (FastAPI + OpenCV) | `5051` | Warp / matrix / draw_boxes / legacy HSV analyze |

```
Browser :5173
    │  /api  (Vite proxy)
    ▼
Backend :5050  ── RTSP grab / ffmpeg stream ──► NVR (e.g. 192.168.1.4)
    │
    ├── TableStore + FrameSource (shared)
    ├── calibration / dataset / model-test routers
    │
    └── HTTP ──► AI worker :5051
                    ├── /calibration/matrix
                    ├── /calibration/warp
                    ├── /vision/draw_boxes
                    └── /analyze  (legacy HSV — debug only)
```

Root scripts (`package.json`):

- `npm run dev:backend`
- `npm run dev:frontend`
- `npm run dev:ai`

---

## 4. Repository layout (important paths)

```
camera/
├── frontend/                 # React UI
│   └── src/
│       ├── App.jsx           # routes / tabs
│       ├── Calibration.jsx   # Phase 1 UI
│       ├── VisionDataset.jsx # optional dataset tool
│       ├── VisionModelTest.jsx # Phase 2B diagnostic UI
│       └── api.js
├── backend/
│   ├── .env                  # secrets (gitignored) — e.g. ROBOFLOW_API_KEY
│   ├── .env.example
│   ├── data/
│   │   ├── tables.json       # Phase 1 calibrations (persisted)
│   │   └── local-credentials.json
│   └── src/
│       ├── index.js          # camera + stream + AI monitor + mounts routers
│       ├── tableStore.js     # calibration persistence + camera_key rebind
│       ├── frameSource.js    # shared RTSP JPEG grabber
│       ├── calibrationRoutes.js
│       ├── datasetRoutes.js
│       ├── modelTestRoutes.js
│       └── detectors/        # DetectorAdapter + RoboflowSnookerDetector
├── ai-worker/
│   ├── app.py
│   ├── calibration.py        # Phase 1 geometry ONLY
│   └── detector.py           # legacy HSV ball detector (not production baseline)
└── datasets/
    ├── scos-v1/              # optional annotated captures
    └── model-tests/          # Phase 2B snapshot dumps
```

---

## 5. Frontend routes / views

| Path | Tab | Purpose |
|------|-----|---------|
| `/` | Cameras | Discover / connect / live RTSP / legacy AI monitor |
| `/table-calibration` | Table calibration | Phase 1 four-corner calibrate + warp preview |
| `/vision-dataset` | Vision dataset | Optional capture + manual bbox annotate + YOLO export |
| `/vision-model-test` | Model test | Diagnostic pretrained detector on warped live feed |

---

## 6. Phase workflows

### Phase 0 — Cameras (done)

1. Load / save NVR credentials.
2. Discover or manually add NVR; **Load all channels**.
3. Connect channel → start live stream (ffmpeg → browser).
4. Legacy **AI Monitor** can still call `/analyze` (HSV) — **not** the production ball detector.

### Phase 1 — Table calibration (done — protect this)

**Goal:** one stable, perspective-corrected playing surface per table.

**Workflow:**

1. Open `/table-calibration`.
2. Select camera channel for the physical table.
3. Create / select table (e.g. Table 1, Table 2).
4. Click four corners on the live snapshot (TL, TR, BR, BL).
5. Backend + AI worker validate corners, compute matrix, save to `tables.json`.
6. Preview warped frame (with margin for cushions/pockets).
7. After backend restart / channel reload, tables **rebind** via `camera_key`.

**Do not put ball detection inside calibration.**

### Phase 2 — Dataset tool (done — optional)

**Goal:** capture warped frames for training / debug; manual boxes if needed.

**Workflow:** `/vision-dataset` → select calibrated table → capture / auto-sample → annotate → export YOLO.

**Policy:** available as a developer tool; **do not require** mass manual annotation as the main path.

### Phase 2B — Pretrained model benchmark (scaffolded — not production)

**Goal:** visually test an existing detector on **real calibrated CCTV**, isolated from HSV / scoring.

**Pipeline:**

```
Calibrated table
  → grab RTSP frame
  → /calibration/warp
  → DetectorAdapter.detect()
  → normalized { class, confidence, x, y, width, height }
  → UI overlay + stats + optional snapshot
```

**UI:** `/vision-model-test`  
**Rate limit:** default **1 FPS** (0.5 / 1 / 2) — never stream full 15 FPS to a paid API.  
**Secrets:** `ROBOFLOW_API_KEY` only in `backend/.env` (never frontend).

**Research outcome (important):**  
Model `snooker-ball-detection-rnhxo-95km5/2` is **not suitable** as the production detector for low-cost commercial SaaS (no clean free weights export; YOLO/AGPL + Roboflow commercial/self-host licensing friction). Keep adapter as optional diagnostic only.

### Future — Production detector (next real AI work)

Intended direction:

- Local weights (ONNX preferred).
- Permissive license (RF-DETR Apache 2.0 or RTMDet MIT).
- Same `DetectorAdapter` interface so the rest of the pipeline stays stable.
- Still **no** scoring / shot / frame-end until detection is trusted on this CCTV.

---

## 7. Detector adapter contract

All future detectors should implement:

```text
getModelInfo() → { id, name, model_id, configured, ... }
detect(jpegBuffer, opts) → {
  detections: [{ class, confidence, x, y, width, height, raw_class? }],
  raw,
  timings_ms,
  image_size
}
```

- `x,y` = **top-left** of box in **warped table pixels**.
- Class names normalize toward SCOS:  
  `red | yellow | green | brown | blue | pink | black | cue_ball`
- Keep detectors **out of** camera connector, calibration, dataset, tracking, scoring.

Registry: `backend/src/detectors/index.js`  
Current: `roboflow-snooker` (diagnostic).

---

## 8. Data & persistence

| File / dir | Purpose |
|------------|---------|
| `backend/data/tables.json` | Phase 1 calibrations |
| `backend/data/local-credentials.json` | NVR username/password (gitignored) |
| `backend/.env` | API keys e.g. Roboflow (gitignored) |
| `datasets/scos-v1/` | Optional vision dataset |
| `datasets/model-tests/` | Benchmark snapshots (frame + boxes + predictions + meta) |

**SCOS ball classes** (`datasets/scos-v1/classes.json`):  
red, yellow, green, brown, blue, pink, black, cue_ball.

---

## 9. Key APIs (backend `/api`)

### Cameras / streams

- `GET/PUT /credentials`
- `GET /cameras`, discover, connect, channels, streams start/stop
- Legacy AI monitor start/stop + events

### Calibration

- `GET/POST /tables`, `PATCH/DELETE /tables/:id`
- `POST /tables/:id/calibrate`
- `POST /tables/:id/preview`
- `GET /tables/:id/diagnostics`
- `GET /cameras/:id/snapshot`

### Dataset

- capture / list / annotate / export under `/dataset/*`

### Model test (diagnostic)

- `GET /model-test/status`
- `GET /model-test/detectors`
- `POST /model-test/infer`
- `POST /model-test/snapshot`
- `GET /model-test/snapshots`

### AI worker

- `POST /calibration/matrix`
- `POST /calibration/warp`
- `POST /vision/draw_boxes`
- `POST /analyze` — **legacy HSV only**

---

## 10. Architectural rules (do not break)

1. **Calibration = geometry only.** No ball colour / contour / scoring inside `calibration.py`.
2. **Warped frame is the vision baseline.** Downstream AI consumes perspective-corrected tables.
3. **Isolate detectors** behind `DetectorAdapter`.
4. **Do not train / score / track / frame-end** until explicitly requested for that phase.
5. **Do not hard-code production to Roboflow hosted API.**
6. **Do not spam cloud APIs** at full CCTV FPS.
7. Keep dataset / annotation tools available but optional.
8. Prefer **local, commercially permissive** models for production.

---

## 11. Decision log (project memory)

| Date / phase | Decision |
|--------------|----------|
| Phase 0 | Local React + Node + ffmpeg live view for Dahua NVR channels |
| Phase 1 | Manual 4-corner calibration with margin; persist via `camera_key` |
| Legacy AI | HSV `/analyze` + rack heuristics kept for debug; **not** production baseline |
| Phase 2 | Dataset capture on warped frames; manual bbox optional |
| Phase 2B | Add Roboflow pretrained adapter **only** for visual benchmark |
| License research | Reject Roboflow Universe snooker model as production detector |
| Next AI | Train/run local RF-DETR or RTMDet (or similar permissive) on real SCOS CCTV |

---

## 12. How to resume work quickly

1. Start backend `:5050`, AI worker `:5051`, frontend `:5173`.
2. Load NVR channels; open **Table calibration** — confirm Table 1 / Table 2 still warp.
3. For diagnostic model test only: set `ROBOFLOW_API_KEY` in `backend/.env`, restart backend, open `/vision-model-test` at low FPS.
4. For production ball AI: **do not** buy into hosted Roboflow for recurring inference — implement local ONNX detector adapter next.

---

*This file is the project memory. Update it when a phase completes or a major architectural decision changes.*
