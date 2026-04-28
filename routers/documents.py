"""
Document routes.

POST/GET/DELETE /workspaces/{id}/documents/*
GET /documents/{id}/raw, /documents/{id}/preview
"""

import asyncio
import logging
import uuid

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Request, UploadFile
from starlette.responses import StreamingResponse

from config import MINIO_BUCKET
from services.storage import get_minio_client
from services.workspace_svc import _ensure_user_workspace

router = APIRouter()
logger = logging.getLogger("meeting-analyzer")


@router.post("/workspaces/{workspace_id}/documents")
async def upload_document(
    request: Request,
    workspace_id: int,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    from main_live import (
        _detect_mime, _serialize_document_row, _extract_and_store,
    )
    await _ensure_user_workspace(request, workspace_id)

    data = await file.read()
    filename = file.filename or "upload"
    mime_type = _detect_mime(filename)
    object_key = f"workspaces/{workspace_id}/documents/{uuid.uuid4()}_{filename}"

    async with get_minio_client(request.app.state.minio_client) as client:
        await client.put_object(
            Bucket=MINIO_BUCKET,
            Key=object_key,
            Body=data,
            ContentType=mime_type,
        )

    async with request.app.state.db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """INSERT INTO documents (workspace_id, filename, object_key, file_size, mime_type)
               VALUES ($1, $2, $3, $4, $5)
               RETURNING id, filename, file_size, mime_type, uploaded_at, executive_summary,
                         key_takeaways, analyzed_at, extracted_text""",
            workspace_id, filename, object_key, len(data), mime_type,
        )
    doc = _serialize_document_row(row)

    background_tasks.add_task(_extract_and_store, doc["id"], workspace_id, data, mime_type, filename)

    return doc


@router.get("/workspaces/{workspace_id}/documents")
async def list_documents(request: Request, workspace_id: int):
    from main_live import _serialize_document_row
    await _ensure_user_workspace(request, workspace_id)
    async with request.app.state.db_pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT d.id, d.filename, d.file_size, d.mime_type, d.uploaded_at, d.extracted_text,
                      d.executive_summary, d.key_takeaways, d.analyzed_at,
                      d.folder_path, d.drive_file_id, d.canvas_file_id, d.object_key, d.preview_pdf_key,
                      (SELECT count(*) FROM document_chunks dc WHERE dc.document_id = d.id AND dc.embedding IS NOT NULL) as embedded_chunks,
                      (SELECT count(*) FROM document_chunks dc WHERE dc.document_id = d.id) as total_chunks
               FROM documents d
               WHERE d.workspace_id = $1
               ORDER BY d.uploaded_at DESC""",
            workspace_id,
        )
    return [_serialize_document_row(r) for r in rows]


@router.get("/workspaces/{workspace_id}/documents/{doc_id}")
async def get_document_detail(request: Request, workspace_id: int, doc_id: int):
    from main_live import _serialize_document_row
    await _ensure_user_workspace(request, workspace_id)
    async with request.app.state.db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """SELECT id, filename, file_size, mime_type, uploaded_at, extracted_text,
                      executive_summary, key_takeaways, analyzed_at, analysis_provider, analysis_model,
                      tables_json, object_key, preview_pdf_key
               FROM documents
               WHERE id = $1 AND workspace_id = $2""",
            doc_id,
            workspace_id,
        )
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")
    return _serialize_document_row(row, include_text=True)


@router.get("/workspaces/{workspace_id}/documents/{doc_id}/download")
async def download_document(request: Request, workspace_id: int, doc_id: int):
    from main_live import _content_disposition
    await _ensure_user_workspace(request, workspace_id)
    async with request.app.state.db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT filename, object_key, mime_type FROM documents WHERE id = $1 AND workspace_id = $2",
            doc_id, workspace_id,
        )
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")

    async with get_minio_client(request.app.state.minio_client) as client:
        response = await client.get_object(Bucket=MINIO_BUCKET, Key=row["object_key"])
        data = await response["Body"].read()

    return StreamingResponse(
        iter([data]),
        media_type=row["mime_type"],
        headers={"Content-Disposition": _content_disposition("attachment", row["filename"])},
    )


@router.get("/documents/{doc_id}/raw")
async def document_raw(doc_id: int, request: Request):
    """Serve raw document file inline for preview with HTTP range request support."""
    from main_live import _content_disposition
    async with request.app.state.db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT filename, object_key, mime_type, drive_file_id FROM documents WHERE id = $1",
            doc_id,
        )
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")
    if not row["object_key"]:
        raise HTTPException(status_code=404, detail="File not stored locally")

    async with get_minio_client(request.app.state.minio_client) as client:
        head = await client.head_object(Bucket=MINIO_BUCKET, Key=row["object_key"])
    total_size: int = head["ContentLength"]

    range_header = request.headers.get("range", "")
    start: int = 0
    end: int = total_size - 1
    status_code = 200
    if range_header and range_header.startswith("bytes="):
        try:
            range_val = range_header[6:]
            s, e = range_val.split("-", 1)
            start = int(s) if s else 0
            end = int(e) if e else total_size - 1
            end = min(end, total_size - 1)
            status_code = 206
        except (ValueError, IndexError):
            pass

    length = end - start + 1
    minio_range = f"bytes={start}-{end}"
    minio_session = request.app.state.minio_client

    async def _stream():
        async with get_minio_client(minio_session) as client:
            resp = await client.get_object(
                Bucket=MINIO_BUCKET,
                Key=row["object_key"],
                Range=minio_range,
            )
            async for chunk in resp["Body"].iter_chunks(chunk_size=65536):
                yield chunk

    headers: dict[str, str] = {
        "Content-Disposition": _content_disposition("inline", row["filename"]),
        "Content-Length": str(length),
        "Accept-Ranges": "bytes",
        "Cache-Control": "public, max-age=3600",
    }
    if status_code == 206:
        headers["Content-Range"] = f"bytes {start}-{end}/{total_size}"

    return StreamingResponse(
        _stream(),
        status_code=status_code,
        media_type=row["mime_type"] or "application/octet-stream",
        headers=headers,
    )


@router.get("/documents/{doc_id}/preview")
async def document_preview_pdf(doc_id: int, request: Request):
    """Serve the generated PDF preview inline with HTTP range request support."""
    async with request.app.state.db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT preview_pdf_key FROM documents WHERE id = $1", doc_id,
        )
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")
    if not row["preview_pdf_key"]:
        raise HTTPException(status_code=404, detail="PDF preview not available")

    async with get_minio_client(request.app.state.minio_client) as client:
        head = await client.head_object(Bucket=MINIO_BUCKET, Key=row["preview_pdf_key"])
    total_size: int = head["ContentLength"]

    range_header = request.headers.get("range", "")
    start: int = 0
    end: int = total_size - 1
    status_code = 200
    if range_header and range_header.startswith("bytes="):
        try:
            range_val = range_header[6:]
            s, e = range_val.split("-", 1)
            start = int(s) if s else 0
            end = int(e) if e else total_size - 1
            end = min(end, total_size - 1)
            status_code = 206
        except (ValueError, IndexError):
            pass

    length = end - start + 1
    minio_range = f"bytes={start}-{end}"
    minio_session = request.app.state.minio_client

    async def _stream():
        async with get_minio_client(minio_session) as client:
            resp = await client.get_object(
                Bucket=MINIO_BUCKET,
                Key=row["preview_pdf_key"],
                Range=minio_range,
            )
            async for chunk in resp["Body"].iter_chunks(chunk_size=65536):
                yield chunk

    headers: dict[str, str] = {
        "Content-Disposition": 'inline; filename="preview.pdf"',
        "Content-Length": str(length),
        "Accept-Ranges": "bytes",
        "Cache-Control": "public, max-age=3600",
    }
    if status_code == 206:
        headers["Content-Range"] = f"bytes {start}-{end}/{total_size}"

    return StreamingResponse(
        _stream(),
        status_code=status_code,
        media_type="application/pdf",
        headers=headers,
    )


@router.delete("/workspaces/{workspace_id}/documents/{doc_id}")
async def delete_document(request: Request, workspace_id: int, doc_id: int):
    await _ensure_user_workspace(request, workspace_id)
    async with request.app.state.db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT object_key, preview_pdf_key FROM documents WHERE id = $1 AND workspace_id = $2",
            doc_id, workspace_id,
        )
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        async with get_minio_client(request.app.state.minio_client) as client:
            for key in [row["object_key"], row["preview_pdf_key"]]:
                if key:
                    try:
                        await client.delete_object(Bucket=MINIO_BUCKET, Key=key)
                    except Exception as e:
                        logger.warning("MinIO delete failed for %s: %s", key, e)
    except Exception as e:
        logger.warning("MinIO unavailable, skipping object cleanup for doc %s: %s", doc_id, e)

    async with request.app.state.db_pool.acquire() as conn:
        await conn.execute("DELETE FROM documents WHERE id = $1", doc_id)

    return {"ok": True}


@router.get("/workspaces/{workspace_id}/documents/{doc_id}/text")
async def get_document_text(request: Request, workspace_id: int, doc_id: int):
    await _ensure_user_workspace(request, workspace_id)
    async with request.app.state.db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT filename, extracted_text FROM documents WHERE id = $1 AND workspace_id = $2",
            doc_id, workspace_id,
        )
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"filename": row["filename"], "text": row["extracted_text"]}


@router.post("/workspaces/{workspace_id}/documents/{doc_id}/analyze")
async def analyze_document_summary(request: Request, workspace_id: int, doc_id: int):
    from main_live import _analyze_document_and_store, _serialize_document_row
    await _ensure_user_workspace(request, workspace_id)
    async with request.app.state.db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """SELECT extracted_text
               FROM documents
               WHERE id = $1 AND workspace_id = $2""",
            doc_id,
            workspace_id,
        )
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")
    extracted_text = row["extracted_text"] or ""
    if not extracted_text.strip():
        raise HTTPException(status_code=400, detail="Document text has not been extracted yet.")
    try:
        await _analyze_document_and_store(doc_id, workspace_id, extracted_text)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Document analysis failed: {exc}") from exc
    return await get_document_detail(request, workspace_id, doc_id)


@router.post("/workspaces/{workspace_id}/documents/{doc_id}/rotate-pages")
async def rotate_document_pages(request: Request, workspace_id: int, doc_id: int, body: dict):
    """Rotate pages in a stored PDF (raw file and/or preview) and overwrite in MinIO."""
    from main_live import (
        _rotate_pdf_pages_sync, _serialize_document_row,
    )
    degrees = body.get("degrees")
    landscape_only = body.get("landscape_only", False)
    page_indices = body.get("page_indices")
    if degrees not in (90, 180, 270):
        raise HTTPException(status_code=400, detail="degrees must be 90, 180, or 270")
    await _ensure_user_workspace(request, workspace_id)
    async with request.app.state.db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT object_key, mime_type, filename, preview_pdf_key FROM documents WHERE id = $1 AND workspace_id = $2",
            doc_id, workspace_id,
        )
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")

    is_pdf = (row["mime_type"] == "application/pdf" or
              (row["filename"] or "").lower().endswith(".pdf"))

    keys_to_rotate = []
    if is_pdf and row["object_key"]:
        keys_to_rotate.append(row["object_key"])
    if row["preview_pdf_key"] and row["preview_pdf_key"] not in keys_to_rotate:
        keys_to_rotate.append(row["preview_pdf_key"])

    if not keys_to_rotate:
        raise HTTPException(status_code=400, detail="No PDF data found for this document.")

    async with get_minio_client(request.app.state.minio_client) as client:
        for key in keys_to_rotate:
            resp = await client.get_object(Bucket=MINIO_BUCKET, Key=key)
            data = await resp["Body"].read()
            rotated = await asyncio.to_thread(
                _rotate_pdf_pages_sync, data, degrees, landscape_only, page_indices
            )
            await client.put_object(
                Bucket=MINIO_BUCKET, Key=key, Body=rotated,
                ContentType="application/pdf",
            )
    return await get_document_detail(request, workspace_id, doc_id)
