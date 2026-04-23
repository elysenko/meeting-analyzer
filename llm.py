"""LLM Runner service client, JSON extraction, and helper utilities."""
import asyncio
import json
import logging
import re
from typing import Any, AsyncIterator

import httpx
from fastapi import HTTPException

logger = logging.getLogger("meeting-analyzer")

def _strip_markdown_fences(text: str) -> str:
    stripped = (text or "").strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _iter_balanced_json_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    stack: list[str] = []
    start: int | None = None
    in_string = False
    escape = False

    for idx, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue

        if char in "{[":
            if not stack:
                start = idx
            stack.append(char)
            continue

        if char not in "}]":
            continue
        if not stack:
            continue

        opener = stack[-1]
        if (opener == "{" and char != "}") or (opener == "[" and char != "]"):
            stack.clear()
            start = None
            continue

        stack.pop()
        if not stack and start is not None:
            candidate = text[start:idx + 1].strip()
            if candidate:
                candidates.append(candidate)
            start = None

    return candidates


def _truncate_for_log(text: str, limit: int = 320) -> str:
    collapsed = " ".join((text or "").split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit].rstrip() + "..."


class _LLMRunnerService:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    @staticmethod
    def _response_detail(resp: httpx.Response, body: bytes | None = None) -> str:
        detail = (body.decode("utf-8", errors="replace") if body is not None else resp.text) or resp.reason_phrase
        try:
            payload = resp.json()
        except Exception:
            payload = None
        if isinstance(payload, dict):
            detail = (
                payload.get("detail")
                or payload.get("error")
                or payload.get("message")
                or detail
            )
        return str(detail)

    async def get_json(self, path: str, *, timeout: float = 30.0) -> Any:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.get(self._url(path))
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code < 500 or attempt >= 2:
                    raise HTTPException(status_code=resp.status_code, detail=self._response_detail(resp))
                last_error = HTTPException(status_code=resp.status_code, detail=self._response_detail(resp))
            except httpx.RequestError as exc:
                last_error = exc
                if attempt >= 2:
                    break
            await asyncio.sleep(0.2 * (attempt + 1))
        if isinstance(last_error, HTTPException):
            raise last_error
        raise HTTPException(status_code=502, detail=f"llm-runner request failed: {last_error}")

    async def post_json(self, path: str, payload: dict[str, Any], *, timeout: float = 3600.0) -> Any:
        last_error: Exception | None = None
        max_attempts = 3
        per_attempt_timeout = timeout if timeout <= 0 else max(10.0, timeout)
        for attempt in range(max_attempts):
            try:
                async with httpx.AsyncClient(timeout=per_attempt_timeout) as client:
                    resp = await client.post(self._url(path), json=payload)
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code < 500 or attempt >= max_attempts - 1:
                    detail = self._response_detail(resp)
                    logger.warning("llm-runner %s %s returned %d: %s", "POST", path, resp.status_code, detail[:500])
                    raise HTTPException(status_code=resp.status_code, detail=detail)
                last_error = HTTPException(status_code=resp.status_code, detail=self._response_detail(resp))
            except httpx.RequestError as exc:
                logger.warning("llm-runner %s %s attempt %d request error (%s): %s", "POST", path, attempt + 1, type(exc).__name__, exc)
                last_error = exc
                if attempt >= max_attempts - 1:
                    break
            await asyncio.sleep(0.2 * (attempt + 1))
        if isinstance(last_error, HTTPException):
            raise last_error
        raise HTTPException(status_code=502, detail=f"llm-runner request failed ({type(last_error).__name__}): {last_error}")

    async def stream_sse_json(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        timeout: httpx.Timeout | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        request_timeout = timeout or httpx.Timeout(connect=10.0, read=3600.0, write=30.0, pool=30.0)
        async with httpx.AsyncClient(timeout=request_timeout) as client:
            async with client.stream("POST", self._url(path), json=payload) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    detail = self._response_detail(resp, body)
                    raise HTTPException(status_code=resp.status_code, detail=detail)
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    try:
                        yield json.loads(line[6:])
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse llm-runner SSE line: %s", line[:200])


llm_runner_service: _LLMRunnerService | None = None


def create_llm_runner_service(base_url: str) -> _LLMRunnerService:
    """Create an LLM Runner service client and set it as the module default."""
    global llm_runner_service
    llm_runner_service = _LLMRunnerService(base_url)
    return llm_runner_service


def _extract_json_payload(text: str) -> Any:
    stripped = _strip_markdown_fences(text)
    candidates: list[str] = []
    seen: set[str] = set()
    for candidate in [stripped] + _iter_balanced_json_candidates(stripped):
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        candidates.append(candidate)
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            continue
    raise ValueError("Model response did not contain valid JSON.")


def _slugify(text: str, max_length: int = 50) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return (slug or "item")[:max_length].strip("-") or "item"


async def _call_llm_runner(
    messages: list[dict[str, Any]],
    *,
    system: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    tools: list[Any] | None = None,
    use_case: str | None = "chat",
    max_tokens: int = 4096,
    timeout: float = 3600.0,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if system:
        payload["system"] = system
    if use_case:
        payload["use_case"] = use_case
    if provider:
        payload["provider"] = provider
    if model:
        payload["model"] = model
    if tools:
        payload["tools"] = tools
    try:
        data = await llm_runner_service.post_json("/v1/chat/turn", payload, timeout=timeout)
    except HTTPException as exc:
        raise RuntimeError(f"llm-runner error: {exc.detail}") from exc
    return {
        "content": data.get("content", ""),
        "provider": data.get("provider"),
        "model": data.get("model"),
        "usage": data.get("usage", {}),
        "response_mode": data.get("response_mode"),
        "skill_name": data.get("skill_name"),
    }


async def _stream_llm_runner(
    messages: list[dict[str, Any]],
    *,
    system: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    tools: list[Any] | None = None,
    use_case: str | None = "chat",
    max_tokens: int = 4096,
):
    payload: dict[str, Any] = {
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if system:
        payload["system"] = system
    if use_case:
        payload["use_case"] = use_case
    if provider:
        payload["provider"] = provider
    if model:
        payload["model"] = model
    if tools:
        payload["tools"] = tools
    timeout = httpx.Timeout(connect=10.0, read=3600.0, write=30.0, pool=30.0)
    try:
        async for event in llm_runner_service.stream_sse_json("/v1/chat/turn/stream", payload, timeout=timeout):
            yield event
    except HTTPException as exc:
        raise RuntimeError(f"llm-runner stream error: {exc.detail}") from exc


async def _call_llm_runner_json(
    messages: list[dict[str, Any]],
    *,
    system: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    use_case: str | None = "chat",
    max_tokens: int = 4096,
    timeout: float = 3600.0,
) -> tuple[Any, dict[str, Any]]:
    def _messages_to_text_block(items: list[dict[str, Any]]) -> str:
        chunks: list[str] = []
        for item in items:
            role = str(item.get("role") or "user").upper()
            content = item.get("content")
            if isinstance(content, list):
                text = "\n".join(
                    str(part.get("text") or "").strip()
                    for part in content
                    if isinstance(part, dict) and part.get("text")
                ).strip()
            else:
                text = str(content or "").strip()
            if text:
                chunks.append(f"{role}:\n{text}")
        return "\n\n".join(chunks)[:18000]

    result = await _call_llm_runner(
        messages,
        system=system,
        provider=provider,
        model=model,
        use_case=use_case,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    try:
        payload = _extract_json_payload(result["content"])
        return payload, {**result, "json_repair_used": False}
    except Exception as exc:
        logger.warning(
            "Failed to parse llm-runner JSON response (provider=%s model=%s use_case=%s): %s",
            result.get("provider") or provider,
            result.get("model") or model,
            use_case,
            _truncate_for_log(result.get("content") or ""),
        )
        repair_prompt = "\n\n".join([
            "The previous model response did not contain valid JSON.",
            "Convert it into valid JSON only.",
            "Do not include markdown fences, commentary, or explanation.",
            ("Original system instructions:\n" + system.strip()) if system else "",
            "Original request:\n" + _messages_to_text_block(messages),
            "Malformed response:\n" + (result.get("content") or ""),
        ]).strip()
        repair_result = await _call_llm_runner(
            [{"role": "user", "content": repair_prompt}],
            system="You repair malformed model output into strict valid JSON. Return JSON only.",
            provider=provider,
            model=model,
            use_case=use_case,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        try:
            payload = _extract_json_payload(repair_result["content"])
            return payload, {
                **repair_result,
                "provider": repair_result.get("provider") or result.get("provider"),
                "model": repair_result.get("model") or result.get("model"),
                "json_repair_used": True,
            }
        except Exception as repair_exc:
            logger.warning(
                "Failed to repair llm-runner JSON response (provider=%s model=%s use_case=%s): %s",
                repair_result.get("provider") or result.get("provider") or provider,
                repair_result.get("model") or result.get("model") or model,
                use_case,
                _truncate_for_log(repair_result.get("content") or ""),
            )
            raise ValueError("Model response did not contain valid JSON after one repair attempt.") from repair_exc

