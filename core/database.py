"""
Database pool initialization.

Provides init_db_pool() which creates the asyncpg connection pool and
runs the schema migration. Called from main_live.py's lifespan function.
"""

import asyncpg

from config import DATABASE_URL
from db_schema import init_db as _init_db_schema


async def init_db_pool() -> asyncpg.Pool:
    """Create the asyncpg connection pool and initialize the DB schema.

    Returns the pool. The caller (lifespan in main_live.py) is responsible
    for storing it and closing it on shutdown.
    """
    return await _init_db_schema(DATABASE_URL)
