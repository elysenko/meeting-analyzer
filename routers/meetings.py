"""
Meeting and todo routes.

GET/DELETE/PATCH /meetings/*, /workspaces/{id}/meetings
GET/POST/PATCH/DELETE /workspaces/{id}/todos/*, /meetings/{id}/todos/{idx}
"""

import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query, Request

from models import (
    MeetingMergeRequest, MeetingWorkspaceUpdate,
    TodoUpdateRequest, WorkspaceTodoCreateRequest, WorkspaceTodoUpdateRequest,
)
from services.todo_svc import _normalize_todo_status, _parse_workspace_todo_id
from services.workspace_svc import _ensure_user_workspace

router = APIRouter()
logger = logging.getLogger("meeting-analyzer")


# ---------------------------------------------------------------------------
# Meeting endpoints
# ---------------------------------------------------------------------------

@router.get("/workspaces/{workspace_id}/meetings")
async def list_workspace_meetings(request: Request, workspace_id: int):
    await _ensure_user_workspace(request, workspace_id)
    async with request.app.state.db_pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT m.id, m.title, m.filename, m.date, m.summary,
                      (SELECT count(*) FROM meeting_chunks mc WHERE mc.meeting_id = m.id AND mc.embedding IS NOT NULL) as embedded_chunks,
                      (SELECT count(*) FROM meeting_chunks mc WHERE mc.meeting_id = m.id) as total_chunks
               FROM meetings m WHERE m.workspace_id = $1 ORDER BY m.date DESC""",
            workspace_id,
        )
        return [dict(r) for r in rows]


@router.get("/meetings")
async def list_meetings(request: Request, unorganized: bool = Query(default=False)):
    uid = getattr(request.state, 'user_id', None)
    async with request.app.state.db_pool.acquire() as conn:
        if uid:
            if unorganized:
                rows = await conn.fetch(
                    "SELECT id, title, filename, date, summary FROM meetings WHERE workspace_id IS NULL AND user_id = $1 ORDER BY date DESC LIMIT 50", uid
                )
            else:
                rows = await conn.fetch(
                    "SELECT id, title, filename, date, summary FROM meetings WHERE user_id = $1 ORDER BY date DESC LIMIT 50", uid
                )
        else:
            if unorganized:
                rows = await conn.fetch(
                    "SELECT id, title, filename, date, summary FROM meetings WHERE workspace_id IS NULL ORDER BY date DESC LIMIT 50"
                )
            else:
                rows = await conn.fetch(
                    "SELECT id, title, filename, date, summary FROM meetings ORDER BY date DESC LIMIT 50"
                )
        return [dict(r) for r in rows]


@router.get("/meetings/{meeting_id}")
async def get_meeting(meeting_id: int, request: Request):
    from main_live import _meeting_todos_payload
    async with request.app.state.db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """SELECT m.id, m.title, m.filename, m.date, m.transcript, m.summary, m.action_items,
                      m.todos, m.email_body, m.workspace_id, w.name AS workspace_name
               FROM meetings m
               LEFT JOIN workspaces w ON w.id = m.workspace_id
               WHERE m.id = $1""",
            meeting_id,
        )
        if not row:
            raise HTTPException(status_code=404, detail="Meeting not found")
        result = dict(row)
        if isinstance(result.get("action_items"), str):
            try:
                result["action_items"] = json.loads(result["action_items"])
            except json.JSONDecodeError:
                result["action_items"] = []
        result["todos"] = _meeting_todos_payload(result)
        return result


@router.delete("/meetings/{meeting_id}")
async def delete_meeting(meeting_id: int, request: Request):
    async with request.app.state.db_pool.acquire() as conn:
        result = await conn.execute("DELETE FROM meetings WHERE id = $1", meeting_id)
    if result == "DELETE 0":
        raise HTTPException(status_code=404, detail="Meeting not found")
    return {"ok": True}


@router.patch("/meetings/{meeting_id}")
async def update_meeting(meeting_id: int, body: dict, request: Request):
    """Rename a meeting."""
    title = (body.get("title") or "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="title is required")
    async with request.app.state.db_pool.acquire() as conn:
        result = await conn.execute(
            "UPDATE meetings SET title = $1 WHERE id = $2",
            title, meeting_id,
        )
    if result == "UPDATE 0":
        raise HTTPException(status_code=404, detail="Meeting not found")
    return {"ok": True, "title": title}


@router.patch("/meetings/{meeting_id}/workspace")
async def move_meeting_to_workspace(request: Request, meeting_id: int, body: MeetingWorkspaceUpdate):
    target_workspace_id = body.workspace_id
    if target_workspace_id is not None:
        await _ensure_user_workspace(request, target_workspace_id)
    async with request.app.state.db_pool.acquire() as conn:
        result = await conn.execute(
            "UPDATE meetings SET workspace_id = $1 WHERE id = $2",
            target_workspace_id,
            meeting_id,
        )
    if result == "UPDATE 0":
        raise HTTPException(status_code=404, detail="Meeting not found")
    return {"ok": True, "workspace_id": target_workspace_id}


@router.post("/meetings/merge")
async def merge_meetings(request: Request, body: MeetingMergeRequest):
    from main_live import _meeting_todos_payload, analyze_with_llm, save_meeting
    if len(body.meeting_ids) < 2:
        raise HTTPException(status_code=400, detail="At least 2 meeting IDs required")
    uid = getattr(request.state, 'user_id', None)
    async with request.app.state.db_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, title, filename, date, transcript, workspace_id, user_id "
            "FROM meetings WHERE id = ANY($1) ORDER BY date ASC",
            body.meeting_ids,
        )
    if len(rows) != len(body.meeting_ids):
        found = {r["id"] for r in rows}
        missing = [mid for mid in body.meeting_ids if mid not in found]
        raise HTTPException(status_code=404, detail=f"Meetings not found: {missing}")
    if uid:
        for r in rows:
            if r["user_id"] and r["user_id"] != uid:
                raise HTTPException(status_code=403, detail=f"Meeting {r['id']} belongs to another user")
    parts = []
    for i, r in enumerate(rows, 1):
        label = f"--- Recording {i} ({r['filename'] or 'unknown'}, {r['date']}) ---"
        parts.append(f"\n\n{label}\n\n{r['transcript'] or ''}")
    combined_transcript = "".join(parts).strip()
    target_ws = body.workspace_id or rows[0]["workspace_id"]
    if target_ws is not None:
        await _ensure_user_workspace(request, target_ws)
    analysis, _meta = await analyze_with_llm(combined_transcript, workspace_id=target_ws)
    merged_filename = f"merged-{len(rows)}-recordings"
    new_id = await save_meeting(merged_filename, combined_transcript, analysis, target_ws, uid)
    if body.delete_originals:
        async with request.app.state.db_pool.acquire() as conn:
            await conn.execute("DELETE FROM meetings WHERE id = ANY($1)", body.meeting_ids)
    async with request.app.state.db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """SELECT m.id, m.title, m.filename, m.date, m.transcript, m.summary, m.action_items,
                      m.todos, m.email_body, m.workspace_id, w.name AS workspace_name
               FROM meetings m
               LEFT JOIN workspaces w ON w.id = m.workspace_id
               WHERE m.id = $1""",
            new_id,
        )
        result = dict(row)
        if isinstance(result.get("action_items"), str):
            try:
                result["action_items"] = json.loads(result["action_items"])
            except json.JSONDecodeError:
                result["action_items"] = []
        result["todos"] = _meeting_todos_payload(result)
        return result


# ---------------------------------------------------------------------------
# Workspace todo endpoints
# ---------------------------------------------------------------------------

@router.get("/workspaces/{workspace_id}/todos")
async def list_workspace_todos(request: Request, workspace_id: int):

    await _ensure_user_workspace(request, workspace_id)
    return await _list_workspace_todo_items(workspace_id)


@router.get("/workspaces/{workspace_id}/todos/{todo_id:path}")
async def get_workspace_todo(request: Request, workspace_id: int, todo_id: str):

    await _ensure_user_workspace(request, workspace_id)
    items = await _list_workspace_todo_items(workspace_id)
    for item in items:
        if item["id"] == todo_id:
            return item
    raise HTTPException(status_code=404, detail="To-do not found")


@router.post("/workspaces/{workspace_id}/todos")
async def create_workspace_todo(request: Request, workspace_id: int, body: WorkspaceTodoCreateRequest):
    from main_live import (
        _normalize_iso_due_date,
        _iso_date_param, _normalize_workspace_manual_todo,
    )
    await _ensure_user_workspace(request, workspace_id)
    task = (body.task or "").strip()
    if not task:
        raise HTTPException(status_code=400, detail="task is required")
    assignee = (body.assignee or "").strip()[:160] or None
    notes = (body.notes or "").strip()[:4000] or None
    status = _normalize_todo_status(body.status)
    due_date = _normalize_iso_due_date(body.due_date, datetime.now(timezone.utc)) if body.due_date else None
    due_date_param = _iso_date_param(due_date)
    async with request.app.state.db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """INSERT INTO workspace_todos (workspace_id, task, assignee, due_date, notes, status, source_type)
               VALUES ($1, $2, $3, $4, $5, $6, 'manual')
               RETURNING id, workspace_id, task, assignee, due_date, notes, status,
                         source_type, source_meeting_id, created_at, updated_at""",
            workspace_id,
            task[:500],
            assignee,
            due_date_param,
            notes,
            status,
        )
    return {"ok": True, "todo": _normalize_workspace_manual_todo(row)}


@router.patch("/workspaces/{workspace_id}/todos/{todo_id:path}")
async def update_workspace_todo(request: Request, workspace_id: int, todo_id: str, body: WorkspaceTodoUpdateRequest):
    from main_live import (
        _normalize_iso_due_date, _iso_date_param, _normalize_workspace_manual_todo,
    )
    await _ensure_user_workspace(request, workspace_id)
    try:
        source, manual_id, _ = _parse_workspace_todo_id(todo_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if source != "manual":
        raise HTTPException(status_code=400, detail="Meeting-derived to-dos must be edited from the meeting to-do route")
    async with request.app.state.db_pool.acquire() as conn:
        existing = await conn.fetchrow(
            """SELECT id, workspace_id, task, assignee, due_date, notes, status,
                      source_type, source_meeting_id, created_at, updated_at
               FROM workspace_todos
               WHERE id = $1 AND workspace_id = $2""",
            manual_id,
            workspace_id,
        )
        if not existing:
            raise HTTPException(status_code=404, detail="To-do not found")
        data = dict(existing)
        provided_fields = set(body.model_fields_set)
        if "task" in provided_fields:
            task = (body.task or "").strip()
            if not task:
                raise HTTPException(status_code=400, detail="task is required")
            data["task"] = task[:500]
        if "assignee" in provided_fields:
            data["assignee"] = (body.assignee or "").strip()[:160] or None
        if "due_date" in provided_fields:
            data["due_date"] = _normalize_iso_due_date(body.due_date, datetime.now(timezone.utc)) if body.due_date else None
        if "notes" in provided_fields:
            data["notes"] = (body.notes or "").strip()[:4000] or None
        if "status" in provided_fields:
            data["status"] = _normalize_todo_status(body.status)
        row = await conn.fetchrow(
            """UPDATE workspace_todos
               SET task = $1, assignee = $2, due_date = $3, notes = $4, status = $5, updated_at = NOW()
               WHERE id = $6 AND workspace_id = $7
               RETURNING id, workspace_id, task, assignee, due_date, notes, status,
                         source_type, source_meeting_id, created_at, updated_at""",
            data["task"],
            data.get("assignee"),
            _iso_date_param(data.get("due_date")),
            data.get("notes"),
            data.get("status"),
            manual_id,
            workspace_id,
        )
    return {"ok": True, "todo": _normalize_workspace_manual_todo(row)}


@router.delete("/workspaces/{workspace_id}/todos/{todo_id:path}")
async def delete_workspace_todo(request: Request, workspace_id: int, todo_id: str):
    await _ensure_user_workspace(request, workspace_id)
    try:
        source, manual_id, _ = _parse_workspace_todo_id(todo_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if source != "manual":
        raise HTTPException(status_code=400, detail="Only manual to-dos can be deleted")
    async with request.app.state.db_pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM workspace_todos WHERE id = $1 AND workspace_id = $2",
            manual_id,
            workspace_id,
        )
        if result == "DELETE 0":
            raise HTTPException(status_code=404, detail="To-do not found")
        session_row = await conn.fetchrow(
            "SELECT id FROM generate_task_sessions WHERE workspace_id = $1 AND todo_id = $2",
            workspace_id, todo_id,
        )
        if session_row:
            await conn.execute(
                "DELETE FROM generate_task_runs WHERE session_id = $1", session_row["id"]
            )
            await conn.execute(
                "DELETE FROM generate_task_sessions WHERE id = $1", session_row["id"]
            )
    return {"ok": True}


# ---------------------------------------------------------------------------
# Meeting todo update (patch single todo item by index)
# ---------------------------------------------------------------------------

@router.patch("/meetings/{meeting_id}/todos/{todo_index}")
async def update_meeting_todo(meeting_id: int, todo_index: int, body: TodoUpdateRequest, request: Request):
    from main_live import (
        _load_meeting_todos_for_update, _ensure_datetime,
        _normalize_iso_due_date,
    )
    meeting, todos = await _load_meeting_todos_for_update(meeting_id)
    if todo_index < 0 or todo_index >= len(todos):
        raise HTTPException(status_code=404, detail="To-do not found")
    reference_dt = _ensure_datetime(meeting.get("date"))
    provided_fields = set(body.model_fields_set)
    if "due_date" in provided_fields:
        todos[todo_index]["due_date"] = _normalize_iso_due_date(body.due_date, reference_dt) if body.due_date else None
    if "assignee" in provided_fields:
        assignee = (body.assignee or "").strip()
        todos[todo_index]["assignee"] = assignee[:160] if assignee else None
    if "status" in provided_fields:
        todos[todo_index]["status"] = _normalize_todo_status(body.status)
    async with request.app.state.db_pool.acquire() as conn:
        await conn.execute(
            "UPDATE meetings SET todos = $1::jsonb WHERE id = $2",
            json.dumps(todos),
            meeting_id,
        )
    return {
        "ok": True,
        "meeting_id": meeting_id,
        "todo_index": todo_index,
        "todo": todos[todo_index],
    }
