"""
Calendar service helpers.

Pure serialization functions — no DB access.
"""


def normalize_calendar_event(row) -> dict:
    """Convert a calendar_events row to API response format."""
    data = dict(row)
    data["start_time"] = data["start_time"].isoformat() if data.get("start_time") else None
    data["end_time"] = data["end_time"].isoformat() if data.get("end_time") else None
    data["created_at"] = data["created_at"].isoformat() if data.get("created_at") else None
    data["updated_at"] = data["updated_at"].isoformat() if data.get("updated_at") else None
    return data
