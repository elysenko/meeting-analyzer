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
