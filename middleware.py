"""
ASGI middleware and request-signing utilities.

Extracted from main_live.py (Phase 2). Imported back by main_live.py.
No imports from the rest of the app besides config.
"""

import hashlib

from config import SESSION_SECRET_KEY, APP_PATH_PREFIX
from fastapi import Request


# ---------------------------------------------------------------------------
# Path-prefix stripping middleware
# ---------------------------------------------------------------------------

class _StripPrefixMiddleware:
    """Strip APP_PATH_PREFIX from request paths when the reverse proxy hasn't done it.

    Nginx rewrites /meeting-analyzer/foo → /foo before forwarding, so the pod
    normally sees only the stripped path. When the app is accessed directly
    (e.g. via a Tailscale port that tunnels straight to the pod), the full
    prefixed path arrives and this middleware strips it so routing works the
    same either way.
    """

    def __init__(self, asgi_app, prefix: str):
        self.app = asgi_app
        self.prefix = prefix.rstrip("/").encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket") and self.prefix:
            raw = scope.get("raw_path", b"")
            if raw == self.prefix or raw.startswith(self.prefix + b"/"):
                stripped = raw[len(self.prefix):] or b"/"
                scope = dict(scope)
                scope["raw_path"] = stripped
                scope["path"] = stripped.decode("latin-1")
        await self.app(scope, receive, send)


# ---------------------------------------------------------------------------
# User-ID signing (prevents X-User-Id header forgery)
# ---------------------------------------------------------------------------

def _sign_user_id(user_id: str) -> str:
    """Sign a user_id with the session secret to prevent header forgery."""
    secret = SESSION_SECRET_KEY or ""
    return hashlib.sha256(f"{user_id}:{secret}".encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Auth middleware
# ---------------------------------------------------------------------------

AUTH_EXEMPT_PREFIXES = ("/health", "/auth/", "/docs", "/openapi.json")


async def auth_middleware(request: Request, call_next):
    path = request.url.path
    if any(path.startswith(p) for p in AUTH_EXEMPT_PREFIXES) or path == "/":
        return await call_next(request)
    # Read user_id from X-User-Id header (injected by page JS from authenticated session)
    user_id = request.headers.get("x-user-id")
    user_sig = request.headers.get("x-user-sig")
    if user_id and user_sig:
        expected_sig = _sign_user_id(user_id)
        if user_sig != expected_sig:
            user_id = None  # signature mismatch — reject
    else:
        user_id = None
    request.state.user_id = user_id
    return await call_next(request)
