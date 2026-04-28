"""
MinIO / object storage helpers.

Provides a context-manager-based MinIO client factory and upload/download/delete
wrappers used by routers and other service modules.

The minio_client session is stored on app.state (as an aiobotocore session).
Callers pass it in — no direct app.state access here so tests can inject mocks.
"""

import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any

from config import MINIO_ENDPOINT, MINIO_BUCKET, MINIO_ACCESS_KEY, MINIO_SECRET_KEY

logger = logging.getLogger("meeting-analyzer")


def get_minio_client(session):
    """Return an async context-manager S3 client from the given aiobotocore session.

    Call as:
        async with get_minio_client(request.app.state.minio_client) as s3:
            ...
    """
    if session is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Document storage not available.")
    return session.create_client(
        "s3",
        endpoint_url=f"http://{MINIO_ENDPOINT}",
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        region_name="us-east-1",
    )


async def upload_bytes(session, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
    """Upload bytes to MinIO and return the object key."""
    async with get_minio_client(session) as client:
        await client.put_object(
            Bucket=MINIO_BUCKET,
            Key=key,
            Body=data,
            ContentType=content_type,
        )
    return key


async def download_bytes(session, key: str) -> bytes:
    """Download and return raw bytes from MinIO."""
    async with get_minio_client(session) as client:
        response = await client.get_object(Bucket=MINIO_BUCKET, Key=key)
        return await response["Body"].read()


async def delete_object(session, key: str) -> None:
    """Delete an object from MinIO."""
    async with get_minio_client(session) as client:
        await client.delete_object(Bucket=MINIO_BUCKET, Key=key)
