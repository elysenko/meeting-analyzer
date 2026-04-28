"""
Workspace, folder, LLM preferences, and user-search routes.

GET/POST /workspaces, /workspaces/{id}/*, /folders/*, /users/search
GET/PUT/POST /llm/models, /settings/llm-defaults, /settings/llm-defaults/apply
"""

import json
import logging

import httpx
from fastapi import APIRouter, HTTPException, Query, Request

from config import (
    KEYCLOAK_CLIENT_ID, KEYCLOAK_CLIENT_SECRET, KEYCLOAK_REALM, KEYCLOAK_URL,
)
from dependencies import get_db_pool
from services.workspace_svc import _ensure_user_workspace, _ensure_workspace_owner
from services.llm_prefs_svc import (
    _fallback_llm_models,
    _get_global_llm_defaults,
    _get_workspace_llm_preferences,
    _merge_llm_preferences,
    _set_global_llm_defaults,
)
from models import (
    ApplyLLMDefaultsRequest, FolderCreate, FolderUpdate,
    WorkspaceCreate, WorkspaceFolderUpdate, WorkspaceLLMPreferences,
    WorkspaceShareRequest, WorkspaceUpdate,
)

router = APIRouter()
logger = logging.getLogger("meeting-analyzer")


# ---------------------------------------------------------------------------
# Internal helpers (imported by other modules via main_live during transition)
# ---------------------------------------------------------------------------

async def _get_keycloak_admin_token() -> str | None:
    """Get a service account token for the Keycloak admin API via client credentials grant."""
    if not KEYCLOAK_CLIENT_SECRET:
        return None
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{KEYCLOAK_URL}/realms/{KEYCLOAK_REALM}/protocol/openid-connect/token",
                data={
                    "grant_type": "client_credentials",
                    "client_id": KEYCLOAK_CLIENT_ID,
                    "client_secret": KEYCLOAK_CLIENT_SECRET,
                },
            )
            if resp.status_code == 200:
                return resp.json().get("access_token")
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# LLM model / settings routes
# ---------------------------------------------------------------------------

@router.get("/llm/models")
async def list_llm_models(request: Request):
    try:
        return await request.app.state.llm_runner_service.get_json("/v1/models", timeout=30.0)
    except Exception as exc:
        logger.warning("llm-runner /v1/models unavailable, using local fallback: %s", exc)
        return _fallback_llm_models()


@router.get("/settings/llm-defaults")
async def get_global_llm_defaults(request: Request):
    return await _get_global_llm_defaults(request.app.state.db_pool)


@router.put("/settings/llm-defaults")
async def put_global_llm_defaults(request: Request, body: WorkspaceLLMPreferences):
    return await _set_global_llm_defaults(request.app.state.db_pool, body.model_dump())


@router.post("/settings/llm-defaults/apply")
async def apply_global_llm_defaults(request: Request, body: ApplyLLMDefaultsRequest):
    scope = (body.scope or "current").strip().lower()
    defaults = await _get_global_llm_defaults(request.app.state.db_pool)
    async with request.app.state.db_pool.acquire() as conn:
        if scope == "all":
            result = await conn.execute(
                "UPDATE workspaces SET llm_preferences = $1::jsonb",
                json.dumps(defaults),
            )
            updated = int(result.split()[-1]) if result.startswith("UPDATE ") else 0
            return {"scope": "all", "updated": updated, "defaults": defaults}
        if scope == "current":
            if not body.workspace_id:
                raise HTTPException(status_code=400, detail="workspace_id is required for current scope")
            result = await conn.execute(
                "UPDATE workspaces SET llm_preferences = $1::jsonb WHERE id = $2",
                json.dumps(defaults),
                body.workspace_id,
            )
            if result == "UPDATE 0":
                raise HTTPException(status_code=404, detail="Workspace not found")
            return {"scope": "current", "updated": 1, "workspace_id": body.workspace_id, "defaults": defaults}
    raise HTTPException(status_code=400, detail="scope must be current or all")


# ---------------------------------------------------------------------------
# Workspace CRUD
# ---------------------------------------------------------------------------

@router.post("/workspaces")
async def create_workspace(request: Request, body: WorkspaceCreate):
    uid = getattr(request.state, 'user_id', None)
    prefs = await _get_global_llm_defaults(request.app.state.db_pool)
    async with request.app.state.db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "INSERT INTO workspaces (name, llm_preferences, folder_id, user_id) VALUES ($1, $2::jsonb, $3, $4) RETURNING id, name, created_at, folder_id",
            body.name,
            json.dumps(prefs),
            body.folder_id,
            uid,
        )
        return dict(row)


@router.get("/workspaces/{workspace_id}/llm-preferences")
async def get_workspace_llm_preferences(request: Request, workspace_id: int):
    await _ensure_user_workspace(request, workspace_id)
    return _merge_llm_preferences(
        await _get_workspace_llm_preferences(request.app.state.db_pool, workspace_id)
    )


@router.put("/workspaces/{workspace_id}/llm-preferences")
async def put_workspace_llm_preferences(request: Request, workspace_id: int, body: WorkspaceLLMPreferences):
    await _ensure_user_workspace(request, workspace_id)
    prefs = _merge_llm_preferences(body.model_dump())
    async with request.app.state.db_pool.acquire() as conn:
        result = await conn.execute(
            "UPDATE workspaces SET llm_preferences = $1::jsonb WHERE id = $2",
            json.dumps(prefs),
            workspace_id,
        )
    if result == "UPDATE 0":
        raise HTTPException(status_code=404, detail="Workspace not found")
    return prefs


@router.get("/workspaces")
async def list_workspaces(request: Request):
    uid = getattr(request.state, 'user_id', None)
    async with request.app.state.db_pool.acquire() as conn:
        if uid:
            rows = await conn.fetch("""
                SELECT w.id, w.name, w.created_at, w.folder_id, COUNT(m.id) AS meeting_count,
                       (w.user_id != $1) AS is_shared
                FROM workspaces w
                LEFT JOIN meetings m ON m.workspace_id = w.id
                LEFT JOIN workspace_shares ws ON ws.workspace_id = w.id AND ws.user_id = $1
                WHERE w.user_id = $1 OR ws.user_id = $1
                GROUP BY w.id
                ORDER BY w.created_at DESC
            """, uid)
        else:
            rows = await conn.fetch("""
                SELECT w.id, w.name, w.created_at, w.folder_id, COUNT(m.id) AS meeting_count,
                       FALSE AS is_shared
                FROM workspaces w
                LEFT JOIN meetings m ON m.workspace_id = w.id
                GROUP BY w.id
                ORDER BY w.created_at DESC
            """)
        return [dict(r) for r in rows]


@router.delete("/workspaces/{workspace_id}")
async def delete_workspace(request: Request, workspace_id: int):
    await _ensure_workspace_owner(request, workspace_id)
    uid = getattr(request.state, 'user_id', None)
    async with request.app.state.db_pool.acquire() as conn:
        if uid:
            await conn.execute("DELETE FROM meetings WHERE workspace_id = $1 AND user_id = $2", workspace_id, uid)
            result = await conn.execute(
                "DELETE FROM workspaces WHERE id = $1 AND user_id = $2", workspace_id, uid
            )
        else:
            await conn.execute("DELETE FROM meetings WHERE workspace_id = $1", workspace_id)
            result = await conn.execute(
                "DELETE FROM workspaces WHERE id = $1", workspace_id
            )
        if result == "DELETE 0":
            raise HTTPException(status_code=404, detail="Workspace not found")
        return {"ok": True}


@router.patch("/workspaces/{workspace_id}")
async def update_workspace(request: Request, workspace_id: int, body: WorkspaceUpdate):
    await _ensure_workspace_owner(request, workspace_id)
    uid = getattr(request.state, 'user_id', None)
    async with request.app.state.db_pool.acquire() as conn:
        if uid:
            result = await conn.execute(
                "UPDATE workspaces SET name = $1 WHERE id = $2 AND user_id = $3",
                body.name, workspace_id, uid)
        else:
            result = await conn.execute(
                "UPDATE workspaces SET name = $1 WHERE id = $2",
                body.name, workspace_id)
        if result == "UPDATE 0":
            raise HTTPException(status_code=404, detail="Workspace not found")
    return {"ok": True}


@router.patch("/workspaces/{workspace_id}/folder")
async def move_workspace_to_folder(request: Request, workspace_id: int, body: WorkspaceFolderUpdate):
    await _ensure_workspace_owner(request, workspace_id)
    uid = getattr(request.state, 'user_id', None)
    async with request.app.state.db_pool.acquire() as conn:
        if body.folder_id is not None:
            if uid:
                folder = await conn.fetchrow("SELECT id FROM workspace_folders WHERE id = $1 AND user_id = $2", body.folder_id, uid)
            else:
                folder = await conn.fetchrow("SELECT id FROM workspace_folders WHERE id = $1", body.folder_id)
            if not folder:
                raise HTTPException(status_code=404, detail="Folder not found")
        if uid:
            result = await conn.execute(
                "UPDATE workspaces SET folder_id = $1 WHERE id = $2 AND user_id = $3", body.folder_id, workspace_id, uid
            )
        else:
            result = await conn.execute(
                "UPDATE workspaces SET folder_id = $1 WHERE id = $2", body.folder_id, workspace_id
            )
        if result == "UPDATE 0":
            raise HTTPException(status_code=404, detail="Workspace not found")
    return {"ok": True}


# ---------------------------------------------------------------------------
# Workspace sharing
# ---------------------------------------------------------------------------

@router.get("/workspaces/{workspace_id}/shares")
async def list_workspace_shares(request: Request, workspace_id: int):
    await _ensure_user_workspace(request, workspace_id)
    async with request.app.state.db_pool.acquire() as conn:
        owner_row = await conn.fetchrow(
            "SELECT w.user_id, u.email, u.name FROM workspaces w LEFT JOIN users u ON u.id = w.user_id WHERE w.id = $1",
            workspace_id,
        )
        members = []
        if owner_row:
            members.append({
                "user_id": owner_row["user_id"],
                "email": owner_row["email"] or "",
                "name": owner_row["name"] or "",
                "is_owner": True,
            })
        rows = await conn.fetch(
            "SELECT ws.user_id, u.email, u.name, ws.created_at FROM workspace_shares ws "
            "JOIN users u ON u.id = ws.user_id WHERE ws.workspace_id = $1 ORDER BY ws.created_at",
            workspace_id,
        )
        for r in rows:
            members.append({
                "user_id": r["user_id"],
                "email": r["email"] or "",
                "name": r["name"] or "",
                "is_owner": False,
            })
    return members


@router.post("/workspaces/{workspace_id}/shares")
async def add_workspace_share(request: Request, workspace_id: int, body: WorkspaceShareRequest):
    await _ensure_workspace_owner(request, workspace_id)
    target_uid: str | None = None
    async with request.app.state.db_pool.acquire() as conn:
        if body.user_id:
            row = await conn.fetchrow("SELECT id FROM users WHERE id = $1", body.user_id)
            if row:
                target_uid = row["id"]
            else:
                target_uid = body.user_id
                await conn.execute(
                    "INSERT INTO users (id, email, name) VALUES ($1, $2, $3) ON CONFLICT (id) DO NOTHING",
                    target_uid, body.email or "", body.name or "",
                )
        elif body.email:
            row = await conn.fetchrow("SELECT id FROM users WHERE LOWER(email) = $1", body.email.strip().lower())
            if row:
                target_uid = row["id"]
        if not target_uid:
            raise HTTPException(status_code=404, detail="User not found")
        owner_uid = await conn.fetchval("SELECT user_id FROM workspaces WHERE id = $1", workspace_id)
        if target_uid == owner_uid:
            raise HTTPException(status_code=400, detail="Cannot share with the workspace owner")
        try:
            await conn.execute(
                "INSERT INTO workspace_shares (workspace_id, user_id) VALUES ($1, $2)",
                workspace_id, target_uid,
            )
        except Exception:
            raise HTTPException(status_code=409, detail="Already shared with this user")
    return {"ok": True, "user_id": target_uid}


@router.delete("/workspaces/{workspace_id}/shares/{target_user_id}")
async def remove_workspace_share(request: Request, workspace_id: int, target_user_id: str):
    await _ensure_workspace_owner(request, workspace_id)
    async with request.app.state.db_pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM workspace_shares WHERE workspace_id = $1 AND user_id = $2",
            workspace_id, target_user_id,
        )
    if result == "DELETE 0":
        raise HTTPException(status_code=404, detail="Share not found")
    return {"ok": True}


# ---------------------------------------------------------------------------
# User search
# ---------------------------------------------------------------------------

@router.get("/users/search")
async def search_users(request: Request, q: str = Query(default="")):
    """Search realm users by name or email. Tries local DB first, falls back to Keycloak admin API."""
    uid = getattr(request.state, 'user_id', None)
    query = q.strip()
    if len(query) < 2:
        return []
    results: list[dict] = []
    seen_ids: set[str] = set()
    async with request.app.state.db_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, email, name FROM users WHERE (LOWER(email) LIKE $1 OR LOWER(name) LIKE $1) AND id != $2 LIMIT 10",
            f"%{query.lower()}%", uid or "",
        )
        for r in rows:
            seen_ids.add(r["id"])
            results.append({"user_id": r["id"], "email": r["email"] or "", "name": r["name"] or ""})
    try:
        admin_token = await _get_keycloak_admin_token()
        if admin_token:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    f"{KEYCLOAK_URL}/admin/realms/{KEYCLOAK_REALM}/users",
                    params={"search": query, "max": 10},
                    headers={"Authorization": f"Bearer {admin_token}"},
                )
                if resp.status_code == 200:
                    for u in resp.json():
                        kid = u.get("id", "")
                        if kid and kid not in seen_ids and kid != uid:
                            seen_ids.add(kid)
                            results.append({
                                "user_id": kid,
                                "email": u.get("email", ""),
                                "name": (u.get("firstName", "") + " " + u.get("lastName", "")).strip() or u.get("username", ""),
                            })
    except Exception:
        pass
    return results[:15]


# ---------------------------------------------------------------------------
# Folder CRUD
# ---------------------------------------------------------------------------

@router.get("/folders")
async def list_folders(request: Request):
    uid = getattr(request.state, 'user_id', None)
    async with request.app.state.db_pool.acquire() as conn:
        if uid:
            rows = await conn.fetch("""
                SELECT f.id, f.name, f.parent_folder_id, f.created_at,
                       COUNT(DISTINCT w.id) AS workspace_count,
                       COUNT(DISTINCT cf.id) AS subfolder_count
                FROM workspace_folders f
                LEFT JOIN workspaces w ON w.folder_id = f.id
                LEFT JOIN workspace_folders cf ON cf.parent_folder_id = f.id
                WHERE f.user_id = $1
                GROUP BY f.id
                ORDER BY f.name
            """, uid)
        else:
            rows = await conn.fetch("""
                SELECT f.id, f.name, f.parent_folder_id, f.created_at,
                       COUNT(DISTINCT w.id) AS workspace_count,
                       COUNT(DISTINCT cf.id) AS subfolder_count
                FROM workspace_folders f
                LEFT JOIN workspaces w ON w.folder_id = f.id
                LEFT JOIN workspace_folders cf ON cf.parent_folder_id = f.id
                GROUP BY f.id
                ORDER BY f.name
            """)
        return [dict(r) for r in rows]


@router.post("/folders")
async def create_folder(request: Request, body: FolderCreate):
    uid = getattr(request.state, 'user_id', None)
    async with request.app.state.db_pool.acquire() as conn:
        if body.parent_folder_id is not None:
            parent = await conn.fetchrow("SELECT id FROM workspace_folders WHERE id = $1", body.parent_folder_id)
            if not parent:
                raise HTTPException(status_code=404, detail="Parent folder not found")
        row = await conn.fetchrow(
            "INSERT INTO workspace_folders (name, parent_folder_id, user_id) VALUES ($1, $2, $3) RETURNING id, name, parent_folder_id, created_at",
            body.name,
            body.parent_folder_id,
            uid,
        )
        return dict(row)


@router.patch("/folders/{folder_id}")
async def update_folder(request: Request, folder_id: int, body: FolderUpdate):
    uid = getattr(request.state, 'user_id', None)
    async with request.app.state.db_pool.acquire() as conn:
        if uid:
            folder = await conn.fetchrow("SELECT id, name, parent_folder_id FROM workspace_folders WHERE id = $1 AND user_id = $2", folder_id, uid)
        else:
            folder = await conn.fetchrow("SELECT id, name, parent_folder_id FROM workspace_folders WHERE id = $1", folder_id)
        if not folder:
            raise HTTPException(status_code=404, detail="Folder not found")
        new_name = body.name if body.name is not None else folder["name"]
        new_parent = folder["parent_folder_id"]
        if body.parent_folder_id != 0:
            new_parent = body.parent_folder_id
            if new_parent is not None:
                if new_parent == folder_id:
                    raise HTTPException(status_code=400, detail="Folder cannot be its own parent")
                check_id = new_parent
                while check_id is not None:
                    ancestor = await conn.fetchrow("SELECT parent_folder_id FROM workspace_folders WHERE id = $1", check_id)
                    if not ancestor:
                        break
                    if ancestor["parent_folder_id"] == folder_id:
                        raise HTTPException(status_code=400, detail="Circular folder reference")
                    check_id = ancestor["parent_folder_id"]
        await conn.execute(
            "UPDATE workspace_folders SET name = $1, parent_folder_id = $2 WHERE id = $3",
            new_name, new_parent, folder_id
        )
        return {"id": folder_id, "name": new_name, "parent_folder_id": new_parent}


@router.delete("/folders/{folder_id}")
async def delete_folder(request: Request, folder_id: int):
    uid = getattr(request.state, 'user_id', None)
    async with request.app.state.db_pool.acquire() as conn:
        if uid:
            result = await conn.execute("DELETE FROM workspace_folders WHERE id = $1 AND user_id = $2", folder_id, uid)
        else:
            result = await conn.execute("DELETE FROM workspace_folders WHERE id = $1", folder_id)
        if result == "DELETE 0":
            raise HTTPException(status_code=404, detail="Folder not found")
    return {"ok": True}
