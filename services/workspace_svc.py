"""
Workspace access-control helpers.

Shared by all routers that need to verify a workspace belongs to the
authenticated user, or that the user is the workspace owner.

These functions accept the FastAPI Request object so they can derive both
the user_id (from request.state) and the DB pool (from request.app.state)
without requiring callers to change their signatures.
"""

from fastapi import HTTPException, Request


async def _ensure_workspace_exists(workspace_id: int, user_id: str | None, pool) -> None:
    """Raise 404 if workspace does not exist or is not accessible to user_id."""
    async with pool.acquire() as conn:
        if user_id:
            row = await conn.fetchrow(
                """SELECT w.id FROM workspaces w
                   WHERE w.id = $1 AND (
                     w.user_id = $2
                     OR EXISTS (SELECT 1 FROM workspace_shares ws WHERE ws.workspace_id = w.id AND ws.user_id = $2)
                   )""",
                workspace_id, user_id,
            )
        else:
            row = await conn.fetchrow("SELECT id FROM workspaces WHERE id = $1", workspace_id)
    if not row:
        raise HTTPException(status_code=404, detail="Workspace not found")


async def _ensure_user_workspace(request: Request, workspace_id: int) -> None:
    """Validate workspace exists AND belongs to the authenticated user (or is shared with them)."""
    uid = getattr(request.state, "user_id", None)
    pool = request.app.state.db_pool
    await _ensure_workspace_exists(workspace_id, uid, pool)


async def _ensure_workspace_owner(request: Request, workspace_id: int) -> None:
    """Validate the authenticated user is the workspace owner."""
    uid = getattr(request.state, "user_id", None)
    if not uid:
        return  # unauthenticated mode
    pool = request.app.state.db_pool
    async with pool.acquire() as conn:
        owner = await conn.fetchval("SELECT user_id FROM workspaces WHERE id = $1", workspace_id)
    if owner != uid:
        raise HTTPException(status_code=403, detail="Owner access required")
