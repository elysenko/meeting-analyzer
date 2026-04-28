"""
FastAPI shared dependencies.

Provides Depends()-compatible callables for injecting shared resources
into route handlers. Importing from here avoids circular imports between
routers — dependencies.py only imports from config and stdlib.
"""

import asyncpg
from fastapi import HTTPException, Request


async def get_db_pool(request: Request) -> asyncpg.Pool:
    """Return the shared asyncpg connection pool from app.state."""
    return request.app.state.db_pool


async def get_current_user(request: Request) -> dict:
    """Return the authenticated user dict from the session.

    Raises HTTP 401 if the session has no user (not logged in via Keycloak).
    """
    user = request.session.get("user")
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user
