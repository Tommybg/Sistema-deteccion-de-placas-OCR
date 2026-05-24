from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from app.config import settings
from app.db.models import Base
from app.db.session import get_engine
from app.inference.loader import load_models
from app.routes import detect, health, history, samples

logging.basicConfig(level=settings.LOG_LEVEL)
log = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address, default_limits=[])


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    log.info("Database schema ready (DATABASE_URL=%s)", settings.DATABASE_URL.split("@")[-1])

    try:
        app.state.models = load_models()
        log.info("Models loaded successfully")
    except Exception as exc:
        log.exception("Failed to load models: %s", exc)
        app.state.models = None

    yield

    app.state.models = None


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS + ([settings.FRONTEND_URL] if settings.FRONTEND_URL not in settings.CORS_ORIGINS else []),
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    log.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


app.include_router(health.router, prefix="/api")
app.include_router(detect.router, prefix="/api")
app.include_router(samples.router, prefix="/api")
app.include_router(history.router, prefix="/api")


# Apply rate limit to the detection endpoints.
limiter.limit(settings.RATE_LIMIT_DETECT)(detect.detect)
limiter.limit(settings.RATE_LIMIT_DETECT)(detect.detect_batch)
