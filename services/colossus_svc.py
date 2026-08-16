"""Colossus integration client — talks to colossus-api's token-authed /integration/v1 edge.

Third workspace provider (after Drive and Canvas): links ONE workspace to ONE Colossus
project and dispatches distilled meeting text, routed server-side by the project's
current_step (spec -> spec-agent message, else -> Plan row).

Auth: the user's account-scoped connect token (colossus_cak_*), minted in Colossus
Settings -> Integrations and stored per-user in user_tokens.colossus_connect_token.
All requests unwrap the standard Colossus {data, error, meta} envelope.
"""

import logging
import os
import uuid

import httpx

logger = logging.getLogger("meeting-analyzer")

# In-cluster default; override for off-cluster deploys.
COLOSSUS_API_URL = os.environ.get(
    "COLOSSUS_API_URL", "http://colossus-api.colossus.svc.cluster.local:3000"
).rstrip("/")

_TIMEOUT = httpx.Timeout(30.0, connect=10.0)


class ColossusError(Exception):
    """Non-2xx from the Colossus integration edge; carries status + detail for HTTP mapping."""

    def __init__(self, status: int, detail: str):
        self.status = status
        self.detail = detail
        super().__init__(f"colossus {status}: {detail}")


def _unwrap(resp: httpx.Response):
    """Raise ColossusError on non-2xx; unwrap the {data, error, meta} envelope."""
    if resp.status_code >= 400:
        try:
            err = resp.json().get("error") or {}
            detail = err.get("message") or resp.text[:300]
        except Exception:
            detail = resp.text[:300]
        raise ColossusError(resp.status_code, detail)
    body = resp.json()
    return body.get("data") if isinstance(body, dict) and "data" in body else body


async def _request(method: str, path: str, token: str, json_body=None, extra_headers=None):
    headers = {"x-colossus-connect-token": token}
    if extra_headers:
        headers.update(extra_headers)
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.request(method, f"{COLOSSUS_API_URL}{path}", json=json_body, headers=headers)
    return _unwrap(resp)


async def list_projects(token: str) -> list[dict]:
    """Owner's projects with {id, name, currentStep, status, linked}. Also validates the token."""
    data = await _request("GET", "/integration/v1/projects", token)
    return data.get("projects", [])


async def create_project(token: str, name: str) -> dict:
    """Create a fresh Colossus project (GitHub repo + provision). Needs GitHub connected in Colossus."""
    data = await _request("POST", "/integration/v1/projects", token, {"name": name})
    return data.get("project", {})


async def link(token: str, deployment_id: str, workspace_id: int, workspace_name: str) -> dict:
    data = await _request(
        "POST", "/integration/v1/link", token,
        {"deploymentId": deployment_id, "workspaceId": str(workspace_id), "workspaceName": workspace_name},
    )
    return data.get("link", {})


async def link_status(token: str, deployment_id: str) -> dict:
    """Live link + project state (bumps lastSeenAt server-side)."""
    return await _request("GET", f"/integration/v1/link/{deployment_id}/status", token)


async def unlink(token: str, deployment_id: str) -> dict:
    return await _request("DELETE", f"/integration/v1/link/{deployment_id}", token)


async def send(token: str, deployment_id: str, current_step: str, title: str, content: str) -> dict:
    """Dispatch distilled text, routed by the project's current_step.

    step == 'spec'  -> spec-agent message (the project is still being specified)
    otherwise       -> new Plan row (an agent/human executes it in Colossus)
    Idempotency-Key is a fresh uuid4 per logical send; callers wanting retry-safety
    across process restarts should pass their own via send_with_key.
    """
    return await send_with_key(token, deployment_id, current_step, title, content, str(uuid.uuid4()))


async def send_with_key(token: str, deployment_id: str, current_step: str, title: str, content: str, idempotency_key: str) -> dict:
    path_kind = "spec-message" if current_step == "spec" else "plans"
    data = await _request(
        "POST", f"/integration/v1/link/{deployment_id}/{path_kind}", token,
        {"title": title, "content": content},
        extra_headers={"Idempotency-Key": idempotency_key},
    )
    data["routed_to"] = "spec" if path_kind == "spec-message" else "plan"
    return data
