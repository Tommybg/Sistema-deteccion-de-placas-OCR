# ANPR Modern Stack (`apps/`)

This directory contains the new production architecture that replaces
the Streamlit prototype with a two-service deployment:

```
apps/
├── api/     FastAPI inference + history backend (Python 3.11)
└── web/     Next.js 15 dashboard frontend (TypeScript)
```

The original Streamlit code in [`app_demo.py`](../app_demo.py) and
[`app_cloud.py`](../app_cloud.py) is left untouched so the existing
Railway deployment keeps working during the cutover.

## Quick start (local Docker)

```bash
# from repo root
docker compose up --build
```

- Web dashboard: <http://localhost:3000>
- API docs:      <http://localhost:8000/api/docs>
- Postgres:      `postgresql://anpr:anpr@localhost:5432/anpr`

## Quick start (native)

```bash
# Backend
cd apps/api
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd ../..
uvicorn app.main:app --reload --port 8000 --app-dir apps/api

# Frontend (new terminal)
cd apps/web
npm install --legacy-peer-deps
cp .env.example .env.local
npm run dev
```

## Architecture at a glance

- **Backend** wraps [`scripts/vehicle_detector.py`](../scripts/vehicle_detector.py),
  [`scripts/brand_detector.py`](../scripts/brand_detector.py), and
  [`scripts/color_classifier.py`](../scripts/color_classifier.py)
  behind a typed REST API. Models load once at startup (FastAPI
  `lifespan`); inference is serialized via an `asyncio.Lock`. Every
  detection is persisted in Postgres (or SQLite locally) for the
  history page and CSV/JSON export.

- **Frontend** is a Next.js 15 App Router app with a dark-themed
  dashboard, drag-drop / webcam / sample inputs, animated bbox overlay
  on `<canvas>`, and a filterable history table. No detection imagery
  is rendered server-side; the browser receives raw bboxes and draws
  them locally for instant overlay-toggle responsiveness.

- **Database**: a single `detections` table with JSONB columns for the
  full vehicle/plate arrays plus promoted `plate_text`/`brand`/`color`
  for index-friendly filtering.

See [`api/README.md`](api/README.md) and [`web/README.md`](web/README.md)
for service-specific docs. The complete deployment runbook lives in
[`../DEPLOYMENT.md`](../DEPLOYMENT.md).
