# Deployment Runbook — ANPR v2 (FastAPI + Next.js on Railway)

This document covers the end-to-end deployment of the new ANPR stack
described in [`apps/README.md`](apps/README.md). The legacy Streamlit
deployment (`app_cloud.py` + root `Dockerfile`) stays live in parallel
during the cutover.

## Topology

```
Railway project: anpr-prod
├── service: api        (Dockerfile = apps/api/Dockerfile)
├── service: web        (Dockerfile = apps/web/Dockerfile)
├── service: postgres   (Railway-managed Postgres plugin)
└── service: streamlit  (existing, eventual deprecation)
```

## One-time setup

1. **Postgres plugin**
   - In Railway: `+ New` → `Database` → `Add PostgreSQL`.
   - Railway exposes a `DATABASE_URL` env var on every service that
     references this plugin. SQLAlchemy needs the `+asyncpg` driver
     suffix — see the env-var note below.

2. **Backend service (`api`)**
   - `+ New` → `GitHub Repo` → pick this repo.
   - **Service name**: `api`
   - **Root directory**: leave blank (build runs from repo root so it
     can copy `scripts/`, `models/`, `samples/` into the image).
   - **Builder**: Dockerfile, path `apps/api/Dockerfile`.
   - **Environment**:
     - `DATABASE_URL` → reference the Postgres plugin variable, but
       prepend `postgresql+asyncpg://` if Railway provides
       `postgresql://`. Use a Railway template variable:
       `${{Postgres.DATABASE_URL}}`. If Railway's value starts with
       `postgresql://`, also set
       `DATABASE_URL_OVERRIDE=postgresql+asyncpg://...` and update
       `app/config.py` accordingly (it currently reads `DATABASE_URL`
       verbatim).
     - `FRONTEND_URL` → public URL of the `web` service
       (e.g. `https://anpr-web.up.railway.app`).
     - `CORS_ORIGINS` → JSON list, e.g. `["https://anpr-web.up.railway.app"]`.
     - `MODEL_DIR=/app/models` (already baked in; override only if you
       mount a Railway Volume).
     - `INFERENCE_DEVICE=cpu`.
     - `CONFIDENCE_DEFAULT=0.5`.
   - **Healthcheck**: `/api/health` with 300s timeout (slow first-request
     model warmup). Railway picks this up from `apps/api/railway.json`.
   - **Expose**: a public domain so the frontend can reach it directly,
     OR keep private and proxy through the web service's API routes.

3. **Frontend service (`web`)**
   - `+ New` → `GitHub Repo` → pick this repo.
   - **Service name**: `web`
   - **Builder**: Dockerfile, path `apps/web/Dockerfile`.
   - **Environment**:
     - `NEXT_PUBLIC_API_URL` → public URL of the `api` service
       (e.g. `https://anpr-api.up.railway.app`).
   - **Expose**: public domain.

4. **DNS / domains**
   - Map `app.your-domain.com` → `web` service.
   - Map `api.your-domain.com` → `api` service.
   - Update `FRONTEND_URL` and `NEXT_PUBLIC_API_URL` after the domains
     are attached.

## Deploy

```bash
git push origin main
```

Railway watches the branch and rebuilds both services on every push.
First build for the backend is ~6–8 minutes (PyTorch + Ultralytics +
model copy). Subsequent builds are 1–2 minutes thanks to Docker layer
caching.

## Post-deploy verification

1. `curl https://api.your-domain.com/api/health` → `{"status":"ok","models_ready":true,...}`
2. `curl https://api.your-domain.com/api/models` → list of loaded models.
3. Visit `https://app.your-domain.com/dashboard`, upload a sample image,
   confirm bboxes render and the right rail populates with plate/brand/color.
4. Visit `https://app.your-domain.com/history`, confirm the detection
   you just ran appears in the table.
5. Export CSV, open in a spreadsheet.

## Database migrations

The first deploy calls `Base.metadata.create_all` during the FastAPI
`lifespan` startup hook (see [`apps/api/app/main.py`](apps/api/app/main.py)).
This is fine for the initial schema, but **switch to Alembic before
making any schema changes** in production:

```bash
cd apps/api
alembic init alembic
# update alembic/env.py to import app.db.models.Base.metadata
alembic revision --autogenerate -m "initial"
alembic upgrade head
```

Wire `alembic upgrade head` into Railway's "Release Command" once a
revision exists.

## Scaling

- The backend pins `--workers 1`. Models are large; one process per
  replica. Scale **horizontally** via Railway replicas instead of
  bumping workers.
- For >5 RPS sustained, consider an upstream queue (Celery / Redis) so
  long-running inference doesn't block the request handler. Today
  inference is serialized by `asyncio.Lock` and the API stays
  responsive thanks to `asyncio.to_thread`.
- GPU acceleration: change the base image to `pytorch/pytorch:2.x-cuda-runtime`
  and set `INFERENCE_DEVICE=cuda`. Requires a Railway GPU tier.

## Cutover plan

1. **Phase 1 — parallel operation** (current state):
   - Streamlit service keeps serving `/`.
   - New `api` + `web` services are deployed on separate URLs.
   - Internal users test the new dashboard for 1–2 weeks.

2. **Phase 2 — DNS swap**:
   - Point the primary domain at the `web` service.
   - Keep the Streamlit service on a legacy subdomain (e.g.
     `streamlit.your-domain.com`) for a 30-day rollback window.

3. **Phase 3 — decommission**:
   - Delete the Streamlit Railway service.
   - Optionally archive `app_demo.py` and `app_cloud.py` to a
     `legacy/` folder (keep them; they're useful for the parity test).

## Rollback

If the new stack misbehaves, point DNS back at the Streamlit service.
No data migration is needed — the new Postgres database is additive.
