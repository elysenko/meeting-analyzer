"""
Core / infrastructure routes.

GET /health, /health/live, /health/ready, /healthz, /favicon.ico, /favicon.svg, GET /
"""

import logging
import pathlib
import os

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

from config import APP_PATH_PREFIX
from dependencies import get_db_pool
from middleware import _sign_user_id

router = APIRouter()
logger = logging.getLogger("meeting-analyzer")

HTML_PAGE = (pathlib.Path(__file__).parent.parent / "static" / "index.html").read_text()

_FAVICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
    '<rect width="32" height="32" rx="6" fill="#1a1714"/>'
    '<path d="M16 5a3.5 3.5 0 0 0-3.5 3.5v7a3.5 3.5 0 0 0 7 0v-7A3.5 3.5 0 0 0 16 5Z" '
    'fill="none" stroke="#a07040" stroke-width="2" stroke-linecap="round"/>'
    '<path d="M22 13v2a6 6 0 0 1-12 0v-2" fill="none" stroke="#a07040" stroke-width="2" stroke-linecap="round"/>'
    '<line x1="16" y1="21" x2="16" y2="25" stroke="#a07040" stroke-width="2" stroke-linecap="round"/>'
    '</svg>'
)


def _inject_team_chat(html_content: str) -> str:
    """Inject the floating team chat widget if configured (currently hidden)."""
    return html_content  # widget hidden


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    user = request.session.get("user")
    if not user:
        return RedirectResponse(url=f"{APP_PATH_PREFIX}/auth/login")

    user_id = user.get("sub") or user.get("id") or ""
    if user_id and request.app.state.db_pool:
        email = user.get("email", "")
        name = user.get("name", user.get("preferred_username", ""))
        try:
            async with request.app.state.db_pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO users (id, email, name, last_login_at) VALUES ($1, $2, $3, NOW()) "
                    "ON CONFLICT (id) DO UPDATE SET email = $2, name = $3, last_login_at = NOW()",
                    user_id, email, name,
                )
        except Exception as _exc:
            logger.warning("Failed to upsert user record for %s: %s", user_id, _exc)
    user_sig = _sign_user_id(user_id) if user_id else ""
    prefix_script = (
        f"<script>const __BASE='{APP_PATH_PREFIX}';"
        f"const __USER_ID='{user_id}';"
        f"const __USER_SIG='{user_sig}';"
        "const __oF=window.fetch;"
        "window.fetch=function(u,o){if(typeof u==='string'&&u.startsWith('/')&&!u.startsWith(__BASE))u=__BASE+u;"
        "o=o||{};o.headers=o.headers||{};"
        "if(__USER_ID){o.headers['X-User-Id']=__USER_ID;o.headers['X-User-Sig']=__USER_SIG};"
        "return __oF.call(this,u,o)};"
        "const __WS=window.WebSocket;"
        "window.WebSocket=function(u,p){let wu=u.replace(/^(wss?:\\/\\/[^\\/]+)(\\/ws\\/)/, '$1'+__BASE+'$2');"
        "if(__USER_ID){const sep=wu.indexOf('?')===-1?'?':'&';"
        "wu+=sep+'_uid='+encodeURIComponent(__USER_ID)+'&_usig='+encodeURIComponent(__USER_SIG)};"
        "return new __WS(wu,p)};"
        "Object.setPrototypeOf(window.WebSocket,__WS);</script>"
    )
    html = HTML_PAGE.replace("</head>", prefix_script + "\n</head>", 1)
    html = _inject_team_chat(html)
    return HTMLResponse(
        html,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/health/live")
async def health_live():
    """Liveness probe — always returns 200 if the process is running."""
    return {"status": "alive"}


@router.get("/health/ready")
async def health_ready(request: Request):
    """Readiness probe — checks database connectivity, returns 503 if down."""
    try:
        async with request.app.state.db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return {"status": "ready"}
    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "error": str(exc)},
        )


@router.get("/healthz")
async def healthz():
    """Legacy health check — kept for backward compatibility."""
    return {"status": "ok", "version": "1.0"}


@router.get("/favicon.ico")
async def favicon():
    return Response(content=_FAVICON_SVG, media_type="image/svg+xml")


@router.get("/favicon.svg")
async def favicon_svg():
    return Response(content=_FAVICON_SVG, media_type="image/svg+xml")
