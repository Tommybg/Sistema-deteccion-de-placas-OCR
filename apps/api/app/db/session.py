from __future__ import annotations

from pathlib import Path
from typing import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import settings


def _ensure_sqlite_parent(url: str) -> None:
    """SQLite needs the parent directory to exist before it can create
    the file. Postgres URLs are no-ops."""
    if not url.startswith("sqlite"):
        return
    # Extract the file path from sqlite+aiosqlite:////absolute/path.db
    # or sqlite+aiosqlite:///relative/path.db
    _, _, after_scheme = url.partition("///")
    if not after_scheme or after_scheme.startswith(":memory:"):
        return
    db_path = Path("/" + after_scheme) if url.startswith("sqlite+aiosqlite:////") else Path(after_scheme)
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass


_ensure_sqlite_parent(settings.DATABASE_URL)

_engine = create_async_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    future=True,
)

_AsyncSessionLocal = async_sessionmaker(
    bind=_engine,
    expire_on_commit=False,
    class_=AsyncSession,
)


def get_engine():
    return _engine


async def get_session() -> AsyncIterator[AsyncSession]:
    async with _AsyncSessionLocal() as session:
        yield session
