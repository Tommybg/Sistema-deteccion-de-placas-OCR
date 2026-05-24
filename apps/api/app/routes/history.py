from __future__ import annotations

import csv
import io
import json
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import and_, delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Detection
from app.db.session import get_session
from app.schemas.detection import DetectionList, DetectionListItem

router = APIRouter(prefix="/history", tags=["history"])


def _filters(plate: Optional[str], brand: Optional[str], color: Optional[str],
             from_dt: Optional[datetime], to_dt: Optional[datetime]):
    clauses = []
    if plate:
        clauses.append(Detection.plate_text.ilike(f"%{plate}%"))
    if brand:
        clauses.append(Detection.brand == brand)
    if color:
        clauses.append(Detection.color == color)
    if from_dt is not None:
        clauses.append(Detection.created_at >= from_dt)
    if to_dt is not None:
        clauses.append(Detection.created_at <= to_dt)
    return and_(*clauses) if clauses else None


@router.get("", response_model=DetectionList)
async def list_history(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    plate: Optional[str] = None,
    brand: Optional[str] = None,
    color: Optional[str] = None,
    from_dt: Optional[datetime] = Query(default=None, alias="from"),
    to_dt: Optional[datetime] = Query(default=None, alias="to"),
    session: AsyncSession = Depends(get_session),
) -> DetectionList:
    where = _filters(plate, brand, color, from_dt, to_dt)

    count_stmt = select(func.count(Detection.id))
    if where is not None:
        count_stmt = count_stmt.where(where)
    total = (await session.execute(count_stmt)).scalar_one()

    stmt = select(Detection).order_by(Detection.created_at.desc()).offset(offset).limit(limit)
    if where is not None:
        stmt = stmt.where(where)
    rows = (await session.execute(stmt)).scalars().all()

    items = [
        DetectionListItem(
            id=row.id,
            created_at=row.created_at,
            source=row.source,
            plate_text=row.plate_text,
            brand=row.brand,
            color=row.color,
            plate_count=len(row.plates or []),
            vehicle_count=len(row.vehicles or []),
            latency_ms=row.latency_ms,
        )
        for row in rows
    ]

    next_offset = offset + len(items)
    next_cursor = str(next_offset) if next_offset < total else None
    return DetectionList(items=items, next_cursor=next_cursor, total=total)


@router.get("/export")
async def export_history(
    format: str = Query("csv", pattern="^(csv|json)$"),
    plate: Optional[str] = None,
    brand: Optional[str] = None,
    color: Optional[str] = None,
    from_dt: Optional[datetime] = Query(default=None, alias="from"),
    to_dt: Optional[datetime] = Query(default=None, alias="to"),
    session: AsyncSession = Depends(get_session),
):
    where = _filters(plate, brand, color, from_dt, to_dt)
    stmt = select(Detection).order_by(Detection.created_at.desc())
    if where is not None:
        stmt = stmt.where(where)
    rows = (await session.execute(stmt)).scalars().all()

    if format == "json":
        payload = [
            {
                "id": r.id,
                "created_at": r.created_at.isoformat(),
                "source": r.source,
                "plate_text": r.plate_text,
                "brand": r.brand,
                "color": r.color,
                "latency_ms": r.latency_ms,
                "confidence_used": r.confidence_used,
                "image_width": r.image_width,
                "image_height": r.image_height,
                "plates": r.plates,
                "vehicles": r.vehicles,
            }
            for r in rows
        ]
        return StreamingResponse(
            iter([json.dumps(payload, indent=2)]),
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=anpr_history.json"},
        )

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "id", "created_at", "source", "plate_text", "brand", "color",
        "latency_ms", "confidence_used", "plate_count", "vehicle_count",
    ])
    for r in rows:
        writer.writerow([
            r.id,
            r.created_at.isoformat(),
            r.source,
            r.plate_text or "",
            r.brand or "",
            r.color or "",
            r.latency_ms,
            r.confidence_used,
            len(r.plates or []),
            len(r.vehicles or []),
        ])
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=anpr_history.csv"},
    )


@router.delete("/{detection_id}")
async def delete_detection(detection_id: str, session: AsyncSession = Depends(get_session)):
    result = await session.execute(delete(Detection).where(Detection.id == detection_id))
    await session.commit()
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Detection not found")
    return {"ok": True}
