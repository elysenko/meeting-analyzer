"""Colossus provider routes — third workspace integration (after Drive, Canvas).

POST   /colossus/connect                       {token}                  validate + store connect token
DELETE /colossus/disconnect                                             remove token
GET    /colossus/status                                                 {connected}
GET    /colossus/projects                                               owner's projects (create-or-pick list)
POST   /colossus/projects                      {name}                   create a new Colossus project
POST   /workspaces/{id}/colossus-link          {deployment_id, project_name}
GET    /workspaces/{id}/colossus-status
DELETE /workspaces/{id}/colossus-link
POST   /workspaces/{id}/colossus-send          {title, content, idempotency_key?}

Auth/DB conventions mirror the Drive/Canvas handlers: request.state.user_id,
_ensure_user_workspace, app.state.db_pool + asyncpg.
"""

import logging

from fastapi import APIRouter, HTTPException, Request

from services import colossus_svc
from services.colossus_svc import ColossusError

router = APIRouter()
logger = logging.getLogger("meeting-analyzer")


def _uid(request: Request) -> str:
    uid = getattr(request.state, "user_id", None)
    if not uid:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return uid


async def _connect_token(request: Request) -> str:
    """The user's stored Colossus connect token, or 400 directing them to Settings."""
    uid = _uid(request)
    from main_live import app  # late import — mirrors integrations.py's main_live coupling
    async with app.state.db_pool.acquire() as conn:
        row = await conn.fetchrow("SELECT colossus_connect_token FROM user_tokens WHERE user_id = $1", uid)
    token = row["colossus_connect_token"] if row and row["colossus_connect_token"] else None
    if not token:
        raise HTTPException(status_code=400, detail="Colossus is not connected — paste a connect token in Settings")
    return token


def _map_colossus_error(exc: ColossusError) -> HTTPException:
    # 401 from Colossus = revoked/invalid token -> tell the user to re-connect (not our session's 401).
    if exc.status == 401:
        return HTTPException(status_code=400, detail="Colossus connect token is invalid or revoked — re-connect in Settings")
    return HTTPException(status_code=exc.status, detail=exc.detail)


@router.post("/colossus/connect")
async def colossus_connect(request: Request, body: dict):
    """Validate a pasted connect token (by listing projects) and store it. Canvas-connect pattern."""
    uid = _uid(request)
    token = (body.get("token") or "").strip()
    if not token.startswith("colossus_cak_"):
        raise HTTPException(status_code=400, detail="That doesn't look like a Colossus connect token (colossus_cak_...)")
    try:
        projects = await colossus_svc.list_projects(token)
    except ColossusError as exc:
        raise _map_colossus_error(exc)
    from main_live import app
    async with app.state.db_pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO user_tokens (user_id, colossus_connect_token, updated_at)
               VALUES ($1, $2, NOW())
               ON CONFLICT (user_id) DO UPDATE SET colossus_connect_token = $2, updated_at = NOW()""",
            uid, token,
        )
    logger.info("Colossus connected for user %s (%d projects visible)", uid, len(projects))
    return {"success": True, "projects": projects}


@router.delete("/colossus/disconnect")
async def colossus_disconnect(request: Request):
    uid = _uid(request)
    from main_live import app
    async with app.state.db_pool.acquire() as conn:
        await conn.execute("UPDATE user_tokens SET colossus_connect_token = NULL, updated_at = NOW() WHERE user_id = $1", uid)
    return {"success": True}


@router.get("/colossus/status")
async def colossus_status(request: Request):
    uid = _uid(request)
    from main_live import app
    async with app.state.db_pool.acquire() as conn:
        row = await conn.fetchrow("SELECT colossus_connect_token FROM user_tokens WHERE user_id = $1", uid)
    return {"connected": bool(row and row["colossus_connect_token"])}


@router.get("/colossus/projects")
async def colossus_projects(request: Request):
    token = await _connect_token(request)
    try:
        return {"projects": await colossus_svc.list_projects(token)}
    except ColossusError as exc:
        raise _map_colossus_error(exc)


@router.post("/colossus/projects")
async def colossus_create_project(request: Request, body: dict):
    """CREATE-path: name a new project; Colossus makes the repo + provisions it."""
    token = await _connect_token(request)
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    try:
        return {"project": await colossus_svc.create_project(token, name)}
    except ColossusError as exc:
        raise _map_colossus_error(exc)


@router.post("/workspaces/{workspace_id}/colossus-link")
async def workspace_colossus_link(request: Request, workspace_id: int, body: dict):
    """CONNECT-path: link this workspace 1:1 to a picked (or just-created) project."""
    from main_live import app, _ensure_user_workspace
    await _ensure_user_workspace(request, workspace_id)
    token = await _connect_token(request)
    deployment_id = (body.get("deployment_id") or "").strip()
    project_name = (body.get("project_name") or "").strip()
    if not deployment_id:
        raise HTTPException(status_code=400, detail="deployment_id is required")
    async with app.state.db_pool.acquire() as conn:
        ws = await conn.fetchrow("SELECT name, colossus_deployment_id FROM workspaces WHERE id = $1", workspace_id)
        if not ws:
            raise HTTPException(status_code=404, detail="Workspace not found")
        if ws["colossus_deployment_id"]:
            raise HTTPException(status_code=409, detail="Workspace is already linked to a Colossus project — unlink first")
    try:
        link = await colossus_svc.link(token, deployment_id, workspace_id, ws["name"] or f"workspace-{workspace_id}")
    except ColossusError as exc:
        raise _map_colossus_error(exc)
    async with app.state.db_pool.acquire() as conn:
        await conn.execute(
            "UPDATE workspaces SET colossus_deployment_id = $1, colossus_project_name = $2, colossus_linked_at = NOW() WHERE id = $3",
            deployment_id, project_name or None, workspace_id,
        )
    return {"ok": True, "link": link}


@router.get("/workspaces/{workspace_id}/colossus-status")
async def workspace_colossus_status(request: Request, workspace_id: int):
    from main_live import app, _ensure_user_workspace
    await _ensure_user_workspace(request, workspace_id)
    async with app.state.db_pool.acquire() as conn:
        ws = await conn.fetchrow(
            "SELECT colossus_deployment_id, colossus_project_name, colossus_linked_at FROM workspaces WHERE id = $1",
            workspace_id,
        )
    if not ws or not ws["colossus_deployment_id"]:
        return {"linked": False}
    out = {
        "linked": True,
        "deployment_id": ws["colossus_deployment_id"],
        "project_name": ws["colossus_project_name"],
        "linked_at": ws["colossus_linked_at"].isoformat() if ws["colossus_linked_at"] else None,
    }
    # Live project state (current_step drives the send routing label) — best-effort.
    try:
        token = await _connect_token(request)
        live = await colossus_svc.link_status(token, ws["colossus_deployment_id"])
        out["project"] = live.get("project")
    except (ColossusError, HTTPException):
        out["project"] = None
    return out


@router.delete("/workspaces/{workspace_id}/colossus-link")
async def workspace_colossus_unlink(request: Request, workspace_id: int):
    from main_live import app, _ensure_user_workspace
    await _ensure_user_workspace(request, workspace_id)
    async with app.state.db_pool.acquire() as conn:
        ws = await conn.fetchrow("SELECT colossus_deployment_id FROM workspaces WHERE id = $1", workspace_id)
    dep = ws["colossus_deployment_id"] if ws else None
    if dep:
        try:
            token = await _connect_token(request)
            await colossus_svc.unlink(token, dep)
        except (ColossusError, HTTPException) as exc:  # remote already unlinked / token gone — clear local anyway
            logger.warning("colossus unlink remote failed (clearing local): %s", exc)
    async with app.state.db_pool.acquire() as conn:
        await conn.execute(
            "UPDATE workspaces SET colossus_deployment_id = NULL, colossus_project_name = NULL, colossus_linked_at = NULL WHERE id = $1",
            workspace_id,
        )
    return {"ok": True}


@router.post("/workspaces/{workspace_id}/colossus-send")
async def workspace_colossus_send(request: Request, workspace_id: int, body: dict):
    """Send distilled text to the linked project — routed by current_step (spec vs plan).

    T4 wires the distill step in front of this; the endpoint itself is a dumb pipe:
    {title, content, idempotency_key?} in, {routed_to, ref, replay} out.
    """
    from main_live import app, _ensure_user_workspace
    await _ensure_user_workspace(request, workspace_id)
    token = await _connect_token(request)
    title = (body.get("title") or "").strip()
    content = (body.get("content") or "").strip()
    if not content:
        raise HTTPException(status_code=400, detail="content is required")
    async with app.state.db_pool.acquire() as conn:
        ws = await conn.fetchrow("SELECT colossus_deployment_id FROM workspaces WHERE id = $1", workspace_id)
    dep = ws["colossus_deployment_id"] if ws else None
    if not dep:
        raise HTTPException(status_code=400, detail="Workspace is not linked to a Colossus project")
    try:
        live = await colossus_svc.link_status(token, dep)
        current_step = (live.get("project") or {}).get("currentStep") or "spec"
        key = (body.get("idempotency_key") or "").strip()
        if key:
            result = await colossus_svc.send_with_key(token, dep, current_step, title, content, key)
        else:
            result = await colossus_svc.send(token, dep, current_step, title, content)
    except ColossusError as exc:
        raise _map_colossus_error(exc)
    return result


@router.post("/workspaces/{workspace_id}/chat/sessions/{session_id}/colossus-send")
async def workspace_chat_colossus_send(request: Request, workspace_id: int, session_id: int):
    """Distill this chat discussion and send it to the linked Colossus project.

    Mirrors the chat->document per-session action (routers/chat.py) but non-streaming:
    load the transcript, distill {title, content} with the workspace's own LLM — phrased
    for the linked project's LIVE currentStep (spec -> specification, else implementation
    plan) — then forward through the existing idempotent /colossus-send pipe.
    Idempotency-Key = sha256(ws:sid:message_count): retrying the same discussion state
    replays the original send instead of duplicating it.
    """
    import hashlib
    from main_live import (
        app, _ensure_user_workspace, _get_workspace_chat_session,
        _list_chat_session_messages, _get_workspace_llm_preferences, _resolve_task_llm,
    )
    from llm import _call_llm_runner_json
    await _ensure_user_workspace(request, workspace_id)
    token = await _connect_token(request)
    async with app.state.db_pool.acquire() as conn:
        ws = await conn.fetchrow(
            "SELECT name, colossus_deployment_id, colossus_project_name FROM workspaces WHERE id = $1",
            workspace_id,
        )
    dep = ws["colossus_deployment_id"] if ws else None
    if not dep:
        raise HTTPException(status_code=400, detail="Workspace is not linked to a Colossus project — link one in Settings first")
    session = await _get_workspace_chat_session(workspace_id, session_id)
    messages = await _list_chat_session_messages(workspace_id, session_id)
    turns = [m for m in messages if (m.get("content") or "").strip()]
    if not turns:
        raise HTTPException(status_code=400, detail="This discussion has no messages to send yet")

    try:
        live = await colossus_svc.link_status(token, dep)
    except ColossusError as exc:
        raise _map_colossus_error(exc)
    current_step = ((live.get("project") or {}).get("currentStep")) or "spec"
    artifact = "product specification" if current_step == "spec" else "implementation plan"

    transcript = "\n".join(f"{m.get('role','user')}: {m.get('content','')}" for m in turns)[-24000:]
    prompt = f"""You are distilling a meeting/workspace discussion into a {artifact} for the software project "{ws['colossus_project_name'] or 'linked project'}".

Discussion transcript (oldest first):
{transcript}

Return ONLY valid JSON with:
- "title": a short, specific title for this {artifact} (<= 12 words)
- "content": the distilled {artifact} in markdown — concrete requirements/decisions/steps from the discussion only; no invented details; keep open questions in a final "Open questions" section.
"""
    preferences = await _get_workspace_llm_preferences(workspace_id)
    provider, model = _resolve_task_llm(preferences, "chat")
    try:
        payload, _meta = await _call_llm_runner_json(
            [{"role": "user", "content": prompt}],
            provider=provider, model=model, use_case="chat",
            max_tokens=2400, timeout=300.0,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Distillation failed: {exc}")
    payload = payload if isinstance(payload, dict) else {}
    title = ((payload.get("title") or session.get("title") or "Meeting discussion")).strip()[:200]
    content = (payload.get("content") or "").strip()
    if not content:
        raise HTTPException(status_code=502, detail="Distillation produced no content — try again")

    key = hashlib.sha256(f"{workspace_id}:{session_id}:{len(messages)}".encode()).hexdigest()
    try:
        result = await colossus_svc.send_with_key(token, dep, current_step, title, content, key)
    except ColossusError as exc:
        raise _map_colossus_error(exc)
    logger.info("Colossus chat-send ws=%s session=%s -> %s (%s, replay=%s)",
                workspace_id, session_id, dep, result.get("routed_to"), result.get("replay"))
    out = dict(result)
    out["title"] = title
    return out
