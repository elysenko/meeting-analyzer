"""
To-do service helpers.

Meeting/manual todo normalisation, status helpers, and workspace todo listing.
All DB-touching functions accept a `pool` parameter (asyncpg pool).
"""

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import HTTPException

from config import (
    TODO_STATUS_ALIASES,
    TODO_STATUS_LABELS,
    TODO_STATUS_OPTIONS,
)

logger = logging.getLogger("meeting-analyzer")

# ---------------------------------------------------------------------------
# Constants (local to this service — not in config.py)
# ---------------------------------------------------------------------------

WEEKDAY_NAMES = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

TODO_ASSIGNEE_FILLER_WORDS = {"and", "&", "the", "of", "for", "to"}
TODO_ASSIGNEE_CONTAINER_WORDS = {
    "team", "group", "committee", "staff", "owners", "owner", "lead", "leads",
    "ops", "sales", "finance", "marketing",
}
TODO_ASSIGNEE_ACTION_STARTERS = {
    "send", "share", "review", "complete", "finish", "draft", "prepare", "follow", "follow-up",
    "coordinate", "create", "write", "call", "email", "update", "build", "deliver", "submit",
    "schedule", "check", "confirm", "reach", "talk", "meet", "analyze", "research",
}


# ---------------------------------------------------------------------------
# Datetime / timestamp helpers
# ---------------------------------------------------------------------------


def ensure_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not value:
        return datetime.now(timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return datetime.now(timezone.utc)


def sortable_timestamp(value: Any) -> float:
    return ensure_datetime(value).timestamp()


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------


def normalize_todo_status(value: Any) -> str:
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    raw = TODO_STATUS_ALIASES.get(raw, raw)
    return raw if raw in TODO_STATUS_OPTIONS else "incomplete"


def todo_status_label(value: Any) -> str:
    return TODO_STATUS_LABELS.get(normalize_todo_status(value), "Incomplete")


# ---------------------------------------------------------------------------
# ID helpers
# ---------------------------------------------------------------------------


def format_workspace_manual_todo_id(todo_id: int) -> str:
    return f"manual:{todo_id}"


def format_workspace_meeting_todo_id(meeting_id: int, todo_index: int) -> str:
    return f"meeting:{meeting_id}:{todo_index}"


def parse_workspace_todo_id(raw_id: str) -> tuple[str, int, int | None]:
    text = str(raw_id or "").strip()
    if not text:
        raise ValueError("Empty to-do id")
    if text.startswith("manual:"):
        return "manual", int(text.split(":", 1)[1]), None
    if text.startswith("meeting:"):
        _, meeting_id, todo_index = text.split(":", 2)
        return "meeting", int(meeting_id), int(todo_index)
    legacy = re.fullmatch(r"(\d+):(\d+)", text)
    if legacy:
        meeting_id = int(legacy.group(1))
        ordinal = int(legacy.group(2))
        return "meeting", meeting_id, max(ordinal - 1, 0)
    raise ValueError("Invalid to-do id")


# ---------------------------------------------------------------------------
# Due-date parsing helpers
# ---------------------------------------------------------------------------


def _normalize_iso_due_date(value: Any, reference_dt: datetime) -> str | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return text
    try:
        import dateutil.parser as date_parser  # noqa: PLC0415

        return date_parser.parse(text, default=reference_dt, fuzzy=True).date().isoformat()
    except (ValueError, TypeError, OverflowError):
        return None


def _parse_due_phrase(phrase: str, reference_dt: datetime) -> str | None:
    lower = (phrase or "").strip().lower()
    if not lower:
        return None
    if "today" in lower:
        return reference_dt.date().isoformat()
    if "tomorrow" in lower:
        return (reference_dt.date() + timedelta(days=1)).isoformat()
    if "next week" in lower:
        return (reference_dt.date() + timedelta(days=7)).isoformat()
    if "end of week" in lower or lower == "eow":
        return (reference_dt.date() + timedelta(days=max(4 - reference_dt.weekday(), 0))).isoformat()
    if "end of month" in lower or lower == "eom":
        month_anchor = reference_dt.replace(day=28) + timedelta(days=4)
        return (month_anchor - timedelta(days=month_anchor.day)).date().isoformat()
    for weekday, index in WEEKDAY_NAMES.items():
        if weekday not in lower:
            continue
        days_ahead = (index - reference_dt.weekday()) % 7
        if lower.startswith("next "):
            days_ahead = (days_ahead or 7) + 7
        elif lower.startswith("this "):
            days_ahead = days_ahead or 0
        elif days_ahead == 0 and reference_dt.hour >= 17:
            days_ahead = 7
        return (reference_dt.date() + timedelta(days=days_ahead)).isoformat()
    try:
        import dateutil.parser as date_parser  # noqa: PLC0415

        return date_parser.parse(phrase, default=reference_dt, fuzzy=True).date().isoformat()
    except (ValueError, TypeError, OverflowError):
        return None


def _extract_due_metadata(text: str, reference_dt: datetime) -> tuple[str | None, str | None]:
    cleaned = (text or "").strip()
    if not cleaned:
        return None, None
    patterns = [
        r"\b(?:due|by|before|on|no later than|complete by|needed by|deliver by)\s+([^.;,\n]+)",
        r"\b(?:tomorrow|today|next week|end of week|eow|end of month|eom)\b",
        r"\b(?:next|this)?\s*(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b",
        r"\b(?:jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december)\s+\d{1,2}(?:,\s*\d{4})?\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if not match:
            continue
        phrase = (match.group(1) if match.lastindex else match.group(0)).strip(" .,:;)")
        due_date = _parse_due_phrase(phrase, reference_dt)
        if due_date:
            return due_date, phrase
    return None, None


# ---------------------------------------------------------------------------
# Assignee inference
# ---------------------------------------------------------------------------


def _looks_like_assignee_prefix(prefix: str) -> bool:
    cleaned = (prefix or "").strip(" -:\t")
    if not cleaned or len(cleaned) > 80:
        return False
    words = [word.strip(" ,.;()") for word in cleaned.split() if word.strip(" ,.;()")]
    if not words or len(words) > 6:
        return False
    if words[0].lower() in TODO_ASSIGNEE_ACTION_STARTERS:
        return False
    for word in words:
        lower = word.lower()
        if lower in TODO_ASSIGNEE_FILLER_WORDS or lower in TODO_ASSIGNEE_CONTAINER_WORDS:
            continue
        if word.isupper():
            continue
        if word[0].isupper():
            continue
        return False
    return True


def infer_todo_assignee(task: str) -> tuple[str | None, str]:
    text = (task or "").strip()
    if not text:
        return None, ""
    for sep in (":", " - ", " – ", " — "):
        if sep not in text:
            continue
        prefix, rest = text.split(sep, 1)
        prefix = prefix.strip()
        rest = rest.strip()
        if rest and _looks_like_assignee_prefix(prefix):
            return prefix[:160], rest[:500]
    match = re.match(
        r"^\s*([A-Z][A-Za-z0-9&'./-]*(?:\s+[A-Z][A-Za-z0-9&'./-]*){0,3}|[A-Za-z]{2,}(?:\s+[A-Za-z]{2,}){0,2}\s+team)\s+to\s+(.+)$",
        text,
    )
    if match and _looks_like_assignee_prefix(match.group(1)):
        return match.group(1).strip()[:160], match.group(2).strip()[:500]
    return None, text[:500]


# ---------------------------------------------------------------------------
# Todo item normalisation
# ---------------------------------------------------------------------------


def normalize_todo_item(item: Any, reference_dt: datetime) -> dict[str, Any] | None:
    if isinstance(item, dict):
        task = str(item.get("task") or item.get("title") or item.get("text") or "").strip()
        assignee = str(item.get("assignee") or "").strip() or None
        due_text = str(item.get("due_text") or item.get("due") or "").strip() or None
        due_date = _normalize_iso_due_date(item.get("due_date"), reference_dt)
    else:
        task = str(item or "").strip()
        assignee = None
        due_text = None
        due_date = None
    if not task:
        return None
    if not assignee:
        inferred_assignee, stripped_task = infer_todo_assignee(task)
        assignee = inferred_assignee or assignee
        task = stripped_task or task
    inferred_due_date, inferred_due_text = _extract_due_metadata(due_text or task, reference_dt)
    return {
        "task": task[:500],
        "assignee": assignee[:160] if assignee else None,
        "due_date": due_date or inferred_due_date,
        "due_text": due_text or inferred_due_text,
    }


def derive_todos_from_action_items(action_items: Any, reference_dt: datetime) -> list[dict[str, Any]]:
    items = action_items
    if isinstance(items, str):
        try:
            items = json.loads(items)
        except json.JSONDecodeError:
            items = [items]
    if not isinstance(items, list):
        return []
    todos: list[dict[str, Any]] = []
    for item in items:
        todo = normalize_todo_item(item, reference_dt)
        if todo:
            todos.append(todo)
    return todos


def normalize_analysis_payload(payload: Any, reference_dt: datetime) -> dict[str, Any]:
    data = payload if isinstance(payload, dict) else {}
    action_items = data.get("action_items") or []
    if isinstance(action_items, str):
        action_items = [action_items]
    action_items = [str(item).strip() for item in action_items if str(item).strip()]

    todos_raw = data.get("todos") or []
    if isinstance(todos_raw, str):
        try:
            todos_raw = json.loads(todos_raw)
        except json.JSONDecodeError:
            todos_raw = []
    todos: list[dict[str, Any]] = []
    if isinstance(todos_raw, list):
        for item in todos_raw:
            todo = normalize_todo_item(item, reference_dt)
            if todo:
                todos.append(todo)
    if not todos and action_items:
        todos = derive_todos_from_action_items(action_items, reference_dt)
    if not action_items and todos:
        action_items = [
            todo["task"] + (f' (Owner: {todo["assignee"]})' if todo.get("assignee") else "")
            for todo in todos
        ]

    return {
        "title": str(data.get("title") or "Untitled Meeting").strip() or "Untitled Meeting",
        "summary": str(data.get("summary") or "").strip(),
        "action_items": action_items,
        "todos": todos,
        "email_body": str(data.get("email_body") or "").strip(),
    }


def meeting_todos_payload(meeting: dict[str, Any]) -> list[dict[str, Any]]:
    reference_dt = ensure_datetime(meeting.get("date"))
    stored = meeting.get("todos")
    if isinstance(stored, str):
        try:
            stored = json.loads(stored)
        except json.JSONDecodeError:
            stored = []
    todos: list[dict[str, Any]] = []
    if isinstance(stored, list):
        for item in stored:
            todo = normalize_todo_item(item, reference_dt)
            if todo:
                todos.append(todo)
    if todos:
        return todos
    return derive_todos_from_action_items(meeting.get("action_items") or [], reference_dt)


def normalize_workspace_manual_todo(row: Any) -> dict[str, Any]:
    item = dict(row)
    due_date = item.get("due_date")
    if due_date is not None and not isinstance(due_date, str):
        due_date = due_date.isoformat()
    created_at = ensure_datetime(item.get("created_at"))
    updated_at = ensure_datetime(item.get("updated_at"))
    meeting_date = item.get("meeting_date")
    return {
        "id": format_workspace_manual_todo_id(int(item["id"])),
        "todo_index": None,
        "task": str(item.get("task") or "").strip(),
        "assignee": str(item.get("assignee") or "").strip() or None,
        "due_date": due_date,
        "due_text": None,
        "meeting_id": item.get("source_meeting_id"),
        "meeting_title": item.get("meeting_title") or "",
        "meeting_date": ensure_datetime(meeting_date) if meeting_date else None,
        "meeting_summary": item.get("meeting_summary") or "",
        "created_at": created_at,
        "updated_at": updated_at,
        "notes": str(item.get("notes") or "").strip(),
        "status": normalize_todo_status(item.get("status")),
        "status_label": todo_status_label(item.get("status")),
        "source_type": "manual",
        "source_label": "Manual",
        "is_manual": True,
    }


def normalize_workspace_meeting_todo(meeting: dict[str, Any], todo: dict[str, Any], todo_index: int) -> dict[str, Any]:
    meeting_dt = ensure_datetime(meeting.get("date"))
    return {
        "id": format_workspace_meeting_todo_id(int(meeting["id"]), int(todo_index)),
        "todo_index": todo_index,
        "task": todo.get("task"),
        "assignee": todo.get("assignee"),
        "due_date": todo.get("due_date"),
        "due_text": todo.get("due_text"),
        "meeting_id": meeting["id"],
        "meeting_title": meeting.get("title") or "",
        "meeting_date": meeting_dt,
        "meeting_summary": meeting.get("summary") or "",
        "created_at": meeting_dt,
        "updated_at": meeting_dt,
        "notes": "",
        "status": normalize_todo_status(todo.get("status")),
        "status_label": todo_status_label(todo.get("status")),
        "source_type": "meeting",
        "source_label": "Meeting",
        "is_manual": False,
    }


def workspace_todo_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    due_date = item.get("due_date")
    timestamp = item.get("updated_at") or item.get("meeting_date") or item.get("created_at")
    return (
        due_date is None,
        due_date or "",
        -sortable_timestamp(timestamp),
        item.get("task") or "",
    )


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


async def load_meeting_todos_for_update(
    pool,
    meeting_id: int,
    *,
    conn: Any = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    owns_conn = conn is None
    if owns_conn:
        conn = await pool.acquire()
    try:
        row = await conn.fetchrow(
            "SELECT id, date, action_items, todos FROM meetings WHERE id = $1",
            meeting_id,
        )
        if not row:
            raise HTTPException(status_code=404, detail="Meeting not found")
        meeting = dict(row)
        if isinstance(meeting.get("action_items"), str):
            try:
                meeting["action_items"] = json.loads(meeting["action_items"])
            except json.JSONDecodeError:
                meeting["action_items"] = []
        todos = meeting_todos_payload(meeting)
        return meeting, todos
    finally:
        if owns_conn and conn is not None:
            await pool.release(conn)


async def set_workspace_todo_status(
    pool,
    workspace_id: int,
    todo_id: str,
    status: str,
    *,
    conn: Any = None,
) -> None:
    normalized_status = normalize_todo_status(status)
    source, owner_id, todo_index = parse_workspace_todo_id(todo_id)
    owns_conn = conn is None
    if owns_conn:
        conn = await pool.acquire()
    if source == "manual":
        try:
            result = await conn.execute(
                """
                UPDATE workspace_todos
                SET status = $1, updated_at = NOW()
                WHERE id = $2 AND workspace_id = $3
                """,
                normalized_status,
                owner_id,
                workspace_id,
            )
            if result == "UPDATE 0":
                raise HTTPException(status_code=404, detail="To-do not found")
            return
        finally:
            if owns_conn and conn is not None:
                await pool.release(conn)
    try:
        meeting, todos = await load_meeting_todos_for_update(pool, owner_id, conn=conn)
        if todo_index is None or todo_index < 0 or todo_index >= len(todos):
            raise HTTPException(status_code=404, detail="To-do not found")
        todos[todo_index]["status"] = normalized_status
        await conn.execute(
            "UPDATE meetings SET todos = $1::jsonb WHERE id = $2 AND workspace_id = $3",
            json.dumps(todos),
            owner_id,
            workspace_id,
        )
    finally:
        if owns_conn and conn is not None:
            await pool.release(conn)


async def list_workspace_todo_items(pool, workspace_id: int) -> list[dict[str, Any]]:
    async with pool.acquire() as conn:
        meeting_rows = await conn.fetch(
            """SELECT id, title, date, summary, action_items, todos
               FROM meetings
               WHERE workspace_id = $1
               ORDER BY date DESC""",
            workspace_id,
        )
        manual_rows = await conn.fetch(
            """SELECT wt.id, wt.workspace_id, wt.task, wt.assignee, wt.due_date, wt.notes, wt.status,
                      wt.source_type, wt.source_meeting_id, wt.created_at, wt.updated_at,
                      m.title AS meeting_title, m.date AS meeting_date, m.summary AS meeting_summary
               FROM workspace_todos wt
               LEFT JOIN meetings m ON m.id = wt.source_meeting_id
               WHERE wt.workspace_id = $1
               ORDER BY wt.updated_at DESC, wt.created_at DESC""",
            workspace_id,
        )

    items: list[dict[str, Any]] = []
    for row in meeting_rows:
        meeting = dict(row)
        if isinstance(meeting.get("action_items"), str):
            try:
                meeting["action_items"] = json.loads(meeting["action_items"])
            except json.JSONDecodeError:
                meeting["action_items"] = []
        for todo_index, todo in enumerate(meeting_todos_payload(meeting)):
            items.append(normalize_workspace_meeting_todo(meeting, todo, todo_index))
    for row in manual_rows:
        items.append(normalize_workspace_manual_todo(row))

    items.sort(key=workspace_todo_sort_key)
    return items
