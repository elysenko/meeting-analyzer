"""
To-do identifier parsing and status normalisation.

Pure functions — no app.state or DB dependencies.
"""

import re
from typing import Any

from config import TODO_STATUS_ALIASES, TODO_STATUS_OPTIONS


def _parse_workspace_todo_id(raw_id: str) -> tuple[str, int, int | None]:
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


def _normalize_todo_status(value: Any) -> str:
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    raw = TODO_STATUS_ALIASES.get(raw, raw)
    return raw if raw in TODO_STATUS_OPTIONS else "incomplete"
