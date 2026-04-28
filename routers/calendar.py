"""
Calendar event routes.

GET/POST/PATCH/DELETE /workspaces/{id}/calendar/*
GET /calendar/all
"""

import logging

from dateutil import parser as date_parser
from fastapi import APIRouter, HTTPException, Request

from models import CalendarEventCreateRequest, CalendarEventUpdateRequest

router = APIRouter()
logger = logging.getLogger("meeting-analyzer")


def _normalize_calendar_event(row) -> dict:
    """Convert a calendar_events row to API response format."""
    data = dict(row)
    data["start_time"] = data["start_time"].isoformat() if data.get("start_time") else None
    data["end_time"] = data["end_time"].isoformat() if data.get("end_time") else None
    data["created_at"] = data["created_at"].isoformat() if data.get("created_at") else None
    data["updated_at"] = data["updated_at"].isoformat() if data.get("updated_at") else None
    return data


@router.get("/workspaces/{workspace_id}/calendar")
async def list_calendar_events(
    request: Request,
    workspace_id: int,
    start: str | None = None,
    end: str | None = None,
):
    """List calendar events for a workspace, optionally filtered by date range."""
    from main_live import _ensure_user_workspace
    await _ensure_user_workspace(request, workspace_id)
    async with request.app.state.db_pool.acquire() as conn:
        if start and end:
            rows = await conn.fetch(
                """SELECT id, workspace_id, title, all_day, is_due_only, start_time, end_time, notes, color, source_type, source_id, created_at, updated_at
                   FROM calendar_events
                   WHERE workspace_id = $1 AND start_time < $3 AND end_time > $2
                   ORDER BY start_time""",
                workspace_id,
                date_parser.parse(start),
                date_parser.parse(end),
            )
        else:
            rows = await conn.fetch(
                """SELECT id, workspace_id, title, all_day, is_due_only, start_time, end_time, notes, color, source_type, source_id, created_at, updated_at
                   FROM calendar_events
                   WHERE workspace_id = $1
                   ORDER BY start_time""",
                workspace_id,
            )
    return [_normalize_calendar_event(r) for r in rows]


@router.post("/workspaces/{workspace_id}/calendar")
async def create_calendar_event(request: Request, workspace_id: int, body: CalendarEventCreateRequest):
    """Create a new calendar event."""
    from main_live import _ensure_user_workspace
    await _ensure_user_workspace(request, workspace_id)
    title = (body.title or "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="title is required")
    try:
        start_time = date_parser.parse(body.start_time)
        end_time = date_parser.parse(body.end_time)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid datetime: {exc}") from exc
    is_due_only = body.is_due_only or False
    if is_due_only:
        end_time = start_time
    elif end_time < start_time:
        raise HTTPException(status_code=400, detail="end_time must be after start_time")
    notes = (body.notes or "").strip()[:4000] or None
    color = (body.color or "").strip()[:20] or None
    async with request.app.state.db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """INSERT INTO calendar_events (workspace_id, title, all_day, is_due_only, start_time, end_time, notes, color)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
               RETURNING id, workspace_id, title, all_day, is_due_only, start_time, end_time, notes, color, created_at, updated_at""",
            workspace_id,
            title[:500],
            body.all_day,
            is_due_only,
            start_time,
            end_time,
            notes,
            color,
        )
    return {"ok": True, "event": _normalize_calendar_event(row)}


@router.get("/workspaces/{workspace_id}/calendar/{event_id}")
async def get_calendar_event(request: Request, workspace_id: int, event_id: int):
    """Get a single calendar event."""
    from main_live import _ensure_user_workspace
    await _ensure_user_workspace(request, workspace_id)
    async with request.app.state.db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """SELECT id, workspace_id, title, all_day, is_due_only, start_time, end_time, notes, color, created_at, updated_at
               FROM calendar_events
               WHERE id = $1 AND workspace_id = $2""",
            event_id,
            workspace_id,
        )
    if not row:
        raise HTTPException(status_code=404, detail="Event not found")
    return _normalize_calendar_event(row)


@router.patch("/workspaces/{workspace_id}/calendar/{event_id}")
async def update_calendar_event(request: Request, workspace_id: int, event_id: int, body: CalendarEventUpdateRequest):
    """Update a calendar event (drag-drop reschedule, edit details)."""
    from main_live import _ensure_user_workspace
    await _ensure_user_workspace(request, workspace_id)
    async with request.app.state.db_pool.acquire() as conn:
        existing = await conn.fetchrow(
            """SELECT id, workspace_id, title, all_day, is_due_only, start_time, end_time, notes, color, created_at, updated_at
               FROM calendar_events
               WHERE id = $1 AND workspace_id = $2""",
            event_id,
            workspace_id,
        )
        if not existing:
            raise HTTPException(status_code=404, detail="Event not found")
        data = dict(existing)
        provided_fields = set(body.model_fields_set)
        if "title" in provided_fields:
            title = (body.title or "").strip()
            if not title:
                raise HTTPException(status_code=400, detail="title is required")
            data["title"] = title[:500]
        if "all_day" in provided_fields:
            data["all_day"] = body.all_day
        if "is_due_only" in provided_fields:
            data["is_due_only"] = body.is_due_only
        if "start_time" in provided_fields:
            try:
                data["start_time"] = date_parser.parse(body.start_time)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Invalid start_time: {exc}") from exc
        if "end_time" in provided_fields:
            try:
                data["end_time"] = date_parser.parse(body.end_time)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Invalid end_time: {exc}") from exc
        if data.get("is_due_only"):
            data["end_time"] = data["start_time"]
        elif data["end_time"] < data["start_time"]:
            raise HTTPException(status_code=400, detail="end_time must be after start_time")
        if "notes" in provided_fields:
            data["notes"] = (body.notes or "").strip()[:4000] or None
        if "color" in provided_fields:
            data["color"] = (body.color or "").strip()[:20] or None
        row = await conn.fetchrow(
            """UPDATE calendar_events
               SET title = $1, all_day = $2, is_due_only = $3, start_time = $4, end_time = $5, notes = $6, color = $7, updated_at = NOW()
               WHERE id = $8 AND workspace_id = $9
               RETURNING id, workspace_id, title, all_day, is_due_only, start_time, end_time, notes, color, created_at, updated_at""",
            data["title"],
            data["all_day"],
            data.get("is_due_only", False),
            data["start_time"],
            data["end_time"],
            data.get("notes"),
            data.get("color"),
            event_id,
            workspace_id,
        )
    return {"ok": True, "event": _normalize_calendar_event(row)}


@router.delete("/workspaces/{workspace_id}/calendar/{event_id}")
async def delete_calendar_event(request: Request, workspace_id: int, event_id: int):
    """Delete a calendar event."""
    from main_live import _ensure_user_workspace
    await _ensure_user_workspace(request, workspace_id)
    async with request.app.state.db_pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM calendar_events WHERE id = $1 AND workspace_id = $2",
            event_id,
            workspace_id,
        )
        if result == "DELETE 0":
            raise HTTPException(status_code=404, detail="Event not found")
    return {"ok": True}


@router.get("/calendar/all")
async def list_all_calendar_events(request: Request, start: str | None = None, end: str | None = None):
    """List calendar events across all workspaces the user has access to."""
    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    async with request.app.state.db_pool.acquire() as conn:
        workspace_rows = await conn.fetch(
            """SELECT DISTINCT ws.id, ws.name
               FROM workspaces ws
               LEFT JOIN workspace_shares sh ON sh.workspace_id = ws.id
               WHERE ws.user_id = $1 OR sh.user_id = $1
               ORDER BY ws.name""",
            user_id,
        )
        workspace_ids = [r["id"] for r in workspace_rows]
        workspace_names = {r["id"]: r["name"] for r in workspace_rows}
        if not workspace_ids:
            return []
        if start and end:
            rows = await conn.fetch(
                """SELECT id, workspace_id, title, all_day, is_due_only, start_time, end_time, notes, color, source_type, source_id, created_at, updated_at
                   FROM calendar_events
                   WHERE workspace_id = ANY($1) AND start_time < $3 AND end_time > $2
                   ORDER BY start_time""",
                workspace_ids,
                date_parser.parse(start),
                date_parser.parse(end),
            )
        else:
            rows = await conn.fetch(
                """SELECT id, workspace_id, title, all_day, is_due_only, start_time, end_time, notes, color, source_type, source_id, created_at, updated_at
                   FROM calendar_events
                   WHERE workspace_id = ANY($1)
                   ORDER BY start_time""",
                workspace_ids,
            )
    events = []
    for r in rows:
        event = _normalize_calendar_event(r)
        event["workspace_name"] = workspace_names.get(r["workspace_id"], "Unknown")
        events.append(event)
    return events
