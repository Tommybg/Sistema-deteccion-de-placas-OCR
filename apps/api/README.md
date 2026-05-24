# ANPR Inference API (FastAPI)

Production backend for the Colombian ANPR system. Wraps the existing
inference code in [scripts/](../../scripts/) behind a typed REST API,
persists every detection in Postgres, and ships as a single Docker
service to Railway.

## Local development

```bash
cd apps/api
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run from the repo root so `scripts.*` imports resolve
cd ../..
uvicorn app.main:app --reload --port 8000 --app-dir apps/api
```

The API will be available at `http://localhost:8000/api/`. Interactive
docs at `http://localhost:8000/api/docs`.

By default, the app writes detections to a local SQLite file
(`apps/api/anpr.db`). Point at Postgres via `DATABASE_URL`:

```bash
export DATABASE_URL="postgresql+asyncpg://user:pass@host:5432/anpr"
```

## REST surface

| Method | Path | Notes |
|--------|------|-------|
| GET    | `/api/health` | Liveness + model-readiness probe |
| GET    | `/api/models` | Loaded model metadata |
| POST   | `/api/detect` | Multipart `image` + `confidence` + `source` |
| POST   | `/api/detect/batch` | Up to 20 images per request |
| GET    | `/api/samples` | List shipped sample images |
| GET    | `/api/samples/{name}` | Serve a sample image |
| GET    | `/api/history` | Paginated, filterable detection log |
| GET    | `/api/history/export?format=csv\|json` | Stream filtered history |
| DELETE | `/api/history/{id}` | Remove one entry |

OpenAPI schema is served from `/api/openapi.json` — the frontend uses
it to generate TypeScript types (`pnpm gen:types` in `apps/web`).

## Parity test (hard gate)

`tests/test_pipeline_parity.py` runs both the legacy `app_demo.detect_plates`
function and the new `app.inference.pipeline.run` against samples and
asserts byte-equivalent output (bboxes ±1 px, exact OCR text, identical
brand/color labels). The migration is only safe to deploy if this test
passes:

```bash
cd apps/api
pytest tests/test_pipeline_parity.py -v
```

Skipped automatically if model weights are missing.
