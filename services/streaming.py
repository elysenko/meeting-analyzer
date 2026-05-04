"""
Async streaming utilities.

Provides keepalive wrappers for long-running coroutines inside SSE and ndjson
streaming responses. Without periodic keepalives, nginx closes the connection
after proxy-read-timeout (180 s) if no bytes are sent.

Pattern — task-drain generator:
    task = asyncio.create_task(some_long_coro())
    async for chunk in keepalive_sse(task, "Still working…"):
        yield chunk
    result = task.result()   # re-raises any task exception
"""

import asyncio
import json
from typing import AsyncGenerator


async def keepalive_sse(
    task: "asyncio.Task",
    msg: str = "Still working…",
    interval: float = 20.0,
) -> AsyncGenerator[str, None]:
    """Yield SSE-format keepalive strings while *task* runs.

    The task is shielded so cancellation of the generator does not cancel it.
    After the generator exits the caller must call ``task.result()`` to
    propagate any exception raised inside the task.
    """
    while not task.done():
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=interval)
        except asyncio.TimeoutError:
            yield f"data: {json.dumps({'type': 'status', 'content': msg}, ensure_ascii=False)}\n\n"


async def keepalive_ndjson(
    task: "asyncio.Task",
    msg: str = "Still working…",
    interval: float = 20.0,
) -> AsyncGenerator[str, None]:
    """Yield ndjson-format keepalive strings while *task* runs.

    Same contract as :func:`keepalive_sse` but uses newline-delimited JSON
    (``{"status": msg}\\n``) instead of SSE framing.
    """
    while not task.done():
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=interval)
        except asyncio.TimeoutError:
            yield json.dumps({"status": msg}) + "\n"
