"""
Database pool initialization.

Provides init_db_pool() which creates the asyncpg connection pool and
runs the schema migration. Called from main_live.py's lifespan function.
"""

import asyncio
import logging

import asyncpg

from config import DATABASE_URL
from db_schema import init_db as _init_db_schema

logger = logging.getLogger("meeting-analyzer")

_DB_RETRY_DELAYS = [2, 4, 8, 16, 32]  # seconds; total up to 62s


async def init_db_pool() -> asyncpg.Pool:
    """Create the asyncpg connection pool and initialize the DB schema.

    Retries with exponential backoff so the worker survives transient startup
    races (e.g. the pod is rescheduled before PostgreSQL is ready).  After all
    retries are exhausted the exception propagates and gunicorn will restart.

    Returns the pool. The caller (lifespan in main_live.py) is responsible
    for storing it and closing it on shutdown.
    """
    last_exc: Exception | None = None
    for attempt, delay in enumerate([0] + _DB_RETRY_DELAYS, start=1):
        if delay:
            logger.warning(
                "DB connection attempt %d failed (%s) — retrying in %ds",
                attempt - 1, last_exc, delay,
            )
            await asyncio.sleep(delay)
        try:
            return await _init_db_schema(DATABASE_URL)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc

    raise RuntimeError(
        f"Could not connect to database after {len(_DB_RETRY_DELAYS) + 1} attempts"
    ) from last_exc
