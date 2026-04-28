"""
LLM provider / preferences helpers.

Pure functions (_fallback_llm_models, _merge_llm_preferences, etc.) have
no app.state dependency.

Async DB functions (_get_global_llm_defaults, _set_global_llm_defaults,
_get_workspace_llm_preferences) accept a pool parameter so callers pass
request.app.state.db_pool — no direct app reference.
"""

import json
import logging
import re
from typing import Any

from fastapi import HTTPException

from config import (
    AGENTIC_LLM_PROVIDERS,
    LLM_PROVIDER_LABELS,
    LLM_RUNNER_CONFIG_PATH,
    LLM_TASK_KEYS,
)

logger = logging.getLogger("meeting-analyzer")


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _read_optional_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""


def _fallback_llm_models() -> dict[str, Any]:
    config_text = _read_optional_text(LLM_RUNNER_CONFIG_PATH)
    if not config_text:
        logger.warning(
            "llm-runner /v1/models unavailable and no local config at %s"
            " — returning empty catalog",
            LLM_RUNNER_CONFIG_PATH,
        )
        return {"providers": [], "default": None}

    default_provider = None
    providers: dict[str, dict[str, Any]] = {}
    in_providers = False
    current_provider = None

    for raw_line in config_text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not raw_line.startswith((" ", "\t")):
            in_providers = line.startswith("providers:")
            current_provider = None
            if line.startswith("default:"):
                default_provider = line.split(":", 1)[1].strip() or None
            continue
        if not in_providers:
            continue
        if re.match(r"^  [A-Za-z0-9_-]+:\s*$", line):
            current_provider = line.strip().rstrip(":")
            providers[current_provider] = {"enabled": False, "models": []}
            continue
        if not current_provider:
            continue
        if not line.startswith("    "):
            continue
        key, _, value = line.strip().partition(":")
        value = value.strip()
        if key == "enabled":
            providers[current_provider]["enabled"] = value.lower() == "true"
        elif key == "default_model":
            providers[current_provider]["default_model"] = value.strip("'\"") or None
        elif key == "models":
            models: list[str] = []
            if value.startswith("[") and value.endswith("]"):
                inner = value[1:-1].strip()
                if inner:
                    models = [item.strip().strip("'\"") for item in inner.split(",")]
            providers[current_provider]["models"] = [model for model in models if model]

    enabled_providers = []
    for provider_id, info in providers.items():
        if not info.get("enabled"):
            continue
        enabled_providers.append({
            "id": provider_id,
            "name": LLM_PROVIDER_LABELS.get(provider_id, provider_id),
            "models": info.get("models") or [],
            "default_model": info.get("default_model"),
            "default": provider_id == default_provider,
            "agentic": provider_id in AGENTIC_LLM_PROVIDERS,
        })

    if not enabled_providers:
        raise HTTPException(
            status_code=502,
            detail="llm-runner model list failed: no enabled providers found",
        )

    return {
        "providers": enabled_providers,
        "default": default_provider or enabled_providers[0]["id"],
        "source": "fallback-config",
    }


def _coerce_legacy_llm_prefs(prefs: dict) -> dict:
    """Migrate old qr/dr/document/pdf/pptx task keys to research/generate."""
    if "research" not in prefs or not any((prefs.get("research") or {}).values()):
        prefs["research"] = prefs.pop("dr", None) or prefs.pop("qr", None) or {}
    else:
        prefs.pop("dr", None)
        prefs.pop("qr", None)
    if "generate" not in prefs or not any((prefs.get("generate") or {}).values()):
        prefs["generate"] = (
            prefs.pop("document", None)
            or prefs.pop("pdf", None)
            or prefs.pop("pptx", None)
            or {}
        )
    else:
        prefs.pop("document", None)
        prefs.pop("pdf", None)
        prefs.pop("pptx", None)
    return prefs


def _merge_llm_preferences(raw: Any) -> dict[str, dict[str, str | None]]:
    merged = {key: {"provider": None, "model": None} for key in LLM_TASK_KEYS}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            raw = {}
    if not isinstance(raw, dict):
        return merged
    raw = _coerce_legacy_llm_prefs(raw)
    for key in LLM_TASK_KEYS:
        value = raw.get(key) or {}
        if isinstance(value, dict):
            merged[key]["provider"] = value.get("provider") or None
            merged[key]["model"] = value.get("model") or None
    return merged


def _deep_copy_llm_preferences(
    prefs: dict[str, dict[str, str | None]],
) -> dict[str, dict[str, str | None]]:
    return json.loads(json.dumps(_merge_llm_preferences(prefs)))


def _default_llm_preferences_from_catalog() -> dict[str, dict[str, str | None]]:
    catalog = _fallback_llm_models()
    providers = catalog.get("providers") or []
    default_provider = next(
        (p for p in providers if p.get("default")), None
    )
    if not default_provider and providers:
        default_provider = providers[0]
    provider_id = (default_provider or {}).get("id")
    model_id = (default_provider or {}).get("default_model")
    if not model_id:
        models = (default_provider or {}).get("models") or []
        model_id = models[0] if models else None
    prefs = _merge_llm_preferences({})
    for task_key in LLM_TASK_KEYS:
        prefs[task_key]["provider"] = provider_id
        prefs[task_key]["model"] = model_id
    return prefs


def _apply_llm_preferences(
    base: dict[str, dict[str, str | None]],
    overrides: dict[str, dict[str, str | None]] | Any,
) -> dict[str, dict[str, str | None]]:
    merged = _deep_copy_llm_preferences(base)
    patch = _merge_llm_preferences(overrides)
    for key in LLM_TASK_KEYS:
        if patch[key].get("provider"):
            merged[key]["provider"] = patch[key]["provider"]
        if patch[key].get("model"):
            merged[key]["model"] = patch[key]["model"]
    return merged


# ---------------------------------------------------------------------------
# Async DB functions — callers pass pool explicitly (no bare app.state)
# ---------------------------------------------------------------------------

async def _get_global_llm_defaults(pool) -> dict[str, dict[str, str | None]]:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT value FROM app_settings WHERE key = 'llm_defaults'"
        )
    if not row:
        return _default_llm_preferences_from_catalog()
    return _merge_llm_preferences(row["value"])


async def _set_global_llm_defaults(
    pool,
    prefs: dict[str, dict[str, str | None]],
) -> dict[str, dict[str, str | None]]:
    merged = _merge_llm_preferences(prefs)
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO app_settings (key, value, updated_at)
            VALUES ('llm_defaults', $1::jsonb, NOW())
            ON CONFLICT (key) DO UPDATE
            SET value = EXCLUDED.value,
                updated_at = NOW()
            """,
            json.dumps(merged),
        )
    return merged


async def _get_workspace_llm_preferences(
    pool,
    workspace_id: int | None,
) -> dict[str, dict[str, str | None]]:
    global_defaults = await _get_global_llm_defaults(pool)
    if not workspace_id:
        return global_defaults
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT llm_preferences FROM workspaces WHERE id = $1",
            workspace_id,
        )
    if not row:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return _apply_llm_preferences(global_defaults, row["llm_preferences"])
