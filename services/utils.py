"""
Generic serialisation / streaming utilities.

These are pure functions with no app.state or DB dependencies.
"""

import json
from typing import Any


def _json_default(value: Any):
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _json_line(value: Any) -> str:
    return json.dumps(value, default=_json_default) + "\n"


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return next((item for item in value if isinstance(item, dict)), {})
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list):
                return next((item for item in parsed if isinstance(item, dict)), {})
            return {}
        except Exception:
            return {}
    return {}


def _coerce_int_list(value: Any) -> list[int]:
    result: list[int] = []
    for item in _json_list(value):
        try:
            result.append(int(item))
        except Exception:
            continue
    return result
