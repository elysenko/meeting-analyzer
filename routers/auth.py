"""
Authentication routes.

GET /auth/login, /auth/callback, /auth/logout, /auth/user
"""

import logging

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse

from config import (
    APP_PATH_PREFIX, KEYCLOAK_CALLBACK_URL, KEYCLOAK_CLIENT_ID,
    KEYCLOAK_CLIENT_SECRET, KEYCLOAK_EXTERNAL_URL, KEYCLOAK_REALM, KEYCLOAK_URL,
)
from oauth_client import oauth

router = APIRouter()
logger = logging.getLogger("meeting-analyzer")


@router.get("/auth/login")
async def auth_login(request: Request, kc_idp_hint: str | None = None):
    """Redirect to Keycloak for authentication."""
    redirect_uri = KEYCLOAK_CALLBACK_URL
    extra_params = {}
    if kc_idp_hint:
        extra_params["kc_idp_hint"] = kc_idp_hint
    return await oauth.keycloak.authorize_redirect(request, redirect_uri, **extra_params)


@router.get("/auth/callback")
async def auth_callback(request: Request):
    """Handle OAuth callback from Keycloak."""
    try:
        token = await oauth.keycloak.authorize_access_token(request)
    except Exception as e:
        if "state" in str(e).lower():
            request.session.clear()
            return RedirectResponse(url=f"{APP_PATH_PREFIX}/auth/login")
        raise HTTPException(status_code=400, detail=f"Authentication failed: {str(e)}")

    user = token.get("userinfo")
    if not user:
        raise HTTPException(status_code=400, detail="Failed to get user info")

    request.session["user"] = dict(user)
    user_id_for_token = user.get("sub") or user.get("id") or ""
    if user_id_for_token and request.app.state.db_pool:
        try:
            async with request.app.state.db_pool.acquire() as conn:
                await conn.execute(
                    """INSERT INTO user_tokens (user_id, access_token, refresh_token, updated_at)
                       VALUES ($1, $2, $3, NOW())
                       ON CONFLICT (user_id) DO UPDATE SET access_token = $2, refresh_token = $3, updated_at = NOW()""",
                    user_id_for_token, token.get("access_token", ""), token.get("refresh_token", ""),
                )
        except Exception as exc:
            logger.warning("Failed to store user tokens: %s", exc)

    user_id = user.get("sub") or user.get("id")
    if user_id:
        email = user.get("email", "")
        name = user.get("name", user.get("preferred_username", ""))
        async with request.app.state.db_pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO users (id, email, name, last_login_at) VALUES ($1, $2, $3, NOW()) "
                "ON CONFLICT (id) DO UPDATE SET email = $2, name = $3, last_login_at = NOW()",
                user_id, email, name,
            )
            orphan_count = await conn.fetchval("SELECT COUNT(*) FROM workspaces WHERE user_id IS NULL")
            if orphan_count and orphan_count > 0:
                await conn.execute("UPDATE workspaces SET user_id = $1 WHERE user_id IS NULL", user_id)
                await conn.execute("UPDATE workspace_folders SET user_id = $1 WHERE user_id IS NULL", user_id)
                await conn.execute("UPDATE meetings SET user_id = $1 WHERE user_id IS NULL", user_id)

    return RedirectResponse(url=f"{APP_PATH_PREFIX}/")


@router.get("/auth/logout")
async def auth_logout(request: Request):
    """Log out user and redirect to Keycloak logout."""
    request.session.clear()

    if KEYCLOAK_EXTERNAL_URL:
        app_url = KEYCLOAK_CALLBACK_URL.rsplit("/auth/", 1)[0] + "/"
        logout_url = (
            f"{KEYCLOAK_EXTERNAL_URL}/realms/{KEYCLOAK_REALM}/protocol/openid-connect/logout"
            f"?post_logout_redirect_uri={app_url}&client_id={KEYCLOAK_CLIENT_ID}"
        )
        return RedirectResponse(url=logout_url)
    return RedirectResponse(url=f"{APP_PATH_PREFIX or '/'}")


@router.get("/auth/user")
async def auth_user(request: Request):
    """Return current user info from session."""
    user = request.session.get("user")
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user
