"""
Text / rendering utilities.

Pure helper functions: XML escaping, markdown rendering, color normalization,
title derivation, and various JSON coercion helpers.
"""

import html
import json
import os
import re
from typing import Any

try:
    import bleach
except Exception:
    bleach = None
try:
    import markdown as py_markdown
except Exception:
    py_markdown = None

from config import HEX_COLOR_RE, MARKDOWN_ALLOWED_TAGS, MARKDOWN_ALLOWED_ATTRIBUTES


# ---------------------------------------------------------------------------
# XML / HTML helpers
# ---------------------------------------------------------------------------


def _esc_xml(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _read_optional_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""


def _render_plaintext_html(text: str) -> str:
    return "<br>".join(html.escape(line) for line in str(text or "").splitlines()) or ""


def _render_markdown_html(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    if py_markdown is None:
        return _render_plaintext_html(raw)
    html_text = py_markdown.markdown(
        raw,
        extensions=["extra", "fenced_code", "tables", "sane_lists", "nl2br"],
        output_format="html5",
    )
    if bleach is None:
        return _render_plaintext_html(raw)
    cleaned = bleach.clean(
        html_text,
        tags=MARKDOWN_ALLOWED_TAGS,
        attributes=MARKDOWN_ALLOWED_ATTRIBUTES,
        protocols=["http", "https", "mailto"],
        strip=True,
    )

    def _set_link_target(attrs, new=False):
        attrs[(None, "target")] = "_blank"
        attrs[(None, "rel")] = "noopener noreferrer"
        return attrs

    return bleach.linkify(cleaned, callbacks=[_set_link_target])


# ---------------------------------------------------------------------------
# JSON coercion helpers
# ---------------------------------------------------------------------------


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


def _coerce_str_list(value: Any) -> list[str]:
    result: list[str] = []
    for item in _json_list(value):
        text = str(item or "").strip()
        if text:
            result.append(text)
    return result


# ---------------------------------------------------------------------------
# Colour utilities
# ---------------------------------------------------------------------------


def _normalize_hex_color(value: str | None) -> str | None:
    if not value:
        return None
    match = HEX_COLOR_RE.search(value)
    if not match:
        return None
    color = match.group(0)
    if len(color) == 4:
        color = "#" + "".join(ch * 2 for ch in color[1:])
    return color.lower()


def _hex_to_rgb(color: str | None) -> tuple[int, int, int]:
    normalized = _normalize_hex_color(color) or "#1a1714"
    return (
        int(normalized[1:3], 16),
        int(normalized[3:5], 16),
        int(normalized[5:7], 16),
    )


def _rgb_to_hex(red: int, green: int, blue: int) -> str:
    return "#" + "".join(f"{max(0, min(255, int(channel))):02x}" for channel in (red, green, blue))


def _derive_brand_secondary_color(primary_color: str | None) -> str:
    red, green, blue = _hex_to_rgb(primary_color)
    brightness = (red * 299 + green * 587 + blue * 114) / 1000
    if brightness >= 150:
        blend_target = 18
        amount = 0.3
    else:
        blend_target = 245
        amount = 0.24
    derived = _rgb_to_hex(
        red + (blend_target - red) * amount,
        green + (blend_target - green) * amount,
        blue + (blend_target - blue) * amount,
    )
    return derived if derived != (_normalize_hex_color(primary_color) or "").lower() else "#8c5a24"


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------


def _excerpt_text(text: Any, limit: int = 1200) -> str:
    value = " ".join(str(text or "").split())
    if not value:
        return ""
    return value[:limit]


def _title_from_filename(filename: str) -> str:
    """Derive a meeting title from the filename by stripping extension and cleaning up."""
    if not filename:
        return "Untitled Meeting"
    name = os.path.splitext(filename)[0]
    name = name.replace("_", " ").replace("-", " ").strip()
    return name if name else "Untitled Meeting"


def _inject_team_chat(html_content: str) -> str:
    """Inject the floating team chat widget if configured (currently hidden)."""
    return html_content  # widget hidden
