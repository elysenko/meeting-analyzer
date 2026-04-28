"""
Document service helpers.

OCR, chunking, embedding pipeline, format conversion, serialization, and backfill helpers.
All DB-touching functions accept a `pool` parameter (asyncpg pool) rather than using app.state.
MinIO operations accept a `minio_session` parameter (request.app.state.minio_client).
"""

import asyncio
import io
import json
import logging
import os
import re
import subprocess
import tempfile
import uuid
from datetime import datetime
from typing import Any

from fastapi import HTTPException

from config import (
    DOCUMENT_CHUNK_OVERLAP_CHARS,
    DOCUMENT_CHUNK_MIN_CHARS,
    DOCUMENT_CHUNK_TARGET_CHARS,
    DOCUMENT_QUERY_TERM_LIMIT,
    DOCUMENT_RETRIEVAL_LIMIT,
    MINIO_BUCKET,
    MINIO_ENDPOINT,
    MINIO_ACCESS_KEY,
    MINIO_SECRET_KEY,
)
from services.embeddings import embed_texts
from services.text_svc import _excerpt_text

logger = logging.getLogger("meeting-analyzer")


# ---------------------------------------------------------------------------
# Text normalisation / chunking
# ---------------------------------------------------------------------------


def _normalize_document_text(text: str | None) -> str:
    cleaned = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def _split_document_into_chunks(text: str | None) -> list[dict[str, Any]]:
    normalized = _normalize_document_text(text)
    if not normalized:
        return []
    chunks: list[dict[str, Any]] = []
    start = 0
    length = len(normalized)
    while start < length:
        hard_end = min(start + DOCUMENT_CHUNK_TARGET_CHARS, length)
        end = hard_end
        if hard_end < length:
            candidate = -1
            search_start = min(start + DOCUMENT_CHUNK_MIN_CHARS, hard_end)
            for marker in ("\n\n", ". ", "? ", "! ", "; ", ": "):
                found = normalized.rfind(marker, search_start, hard_end)
                if found >= 0:
                    candidate = max(candidate, found + len(marker))
            if candidate > start:
                end = candidate
        if end <= start:
            end = hard_end
        content = normalized[start:end].strip()
        if content:
            chunks.append({
                "chunk_index": len(chunks),
                "char_start": start,
                "char_end": end,
                "content": content,
            })
        if end >= length:
            break
        start = max(end - DOCUMENT_CHUNK_OVERLAP_CHARS, start + 1)
        while start < length and normalized[start].isspace():
            start += 1
    return chunks


# ---------------------------------------------------------------------------
# Document chunk DB helpers
# ---------------------------------------------------------------------------


async def replace_document_chunks(pool, document_id: int, workspace_id: int, extracted_text: str | None) -> None:
    chunks = _split_document_into_chunks(extracted_text)
    if not chunks:
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM document_chunks WHERE document_id = $1", document_id)
        return

    texts = [str(chunk["content"]) for chunk in chunks]
    try:
        embeddings = embed_texts(texts)
    except Exception as exc:
        logger.warning("Embedding generation failed for document %s: %s", document_id, exc)
        embeddings = [None] * len(chunks)

    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute("DELETE FROM document_chunks WHERE document_id = $1", document_id)
            for chunk, emb in zip(chunks, embeddings):
                await conn.execute(
                    """
                    INSERT INTO document_chunks (
                        workspace_id, document_id, chunk_index, char_start, char_end, content, search_vector, embedding
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, to_tsvector('english', $6), $7)
                    ON CONFLICT (document_id, chunk_index) DO UPDATE
                    SET workspace_id = EXCLUDED.workspace_id,
                        char_start = EXCLUDED.char_start,
                        char_end = EXCLUDED.char_end,
                        content = EXCLUDED.content,
                        search_vector = EXCLUDED.search_vector,
                        embedding = EXCLUDED.embedding
                    """,
                    workspace_id,
                    document_id,
                    int(chunk["chunk_index"]),
                    int(chunk["char_start"]),
                    int(chunk["char_end"]),
                    str(chunk["content"]),
                    str(emb) if emb else None,
                )
    logger.info("Chunked+embedded document %s: %d chunks", document_id, len(chunks))


async def ensure_document_chunks(pool, workspace_id: int, document_ids: list[int]) -> None:
    target_ids = [int(item) for item in document_ids if item]
    if not target_ids:
        return
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT d.id, d.extracted_text
            FROM documents d
            WHERE d.workspace_id = $1
              AND d.id = ANY($2::int[])
              AND COALESCE(d.extracted_text, '') <> ''
              AND NOT EXISTS (
                    SELECT 1
                    FROM document_chunks dc
                    WHERE dc.document_id = d.id
                )
            """,
            workspace_id,
            target_ids,
        )
    for row in rows:
        try:
            await replace_document_chunks(pool, row["id"], workspace_id, row["extracted_text"])
        except Exception as exc:
            logger.warning("Document chunk backfill failed for doc %s: %s", row["id"], exc)


# ---------------------------------------------------------------------------
# Meeting chunk DB helpers
# ---------------------------------------------------------------------------


async def replace_meeting_chunks(pool, meeting_id: int, workspace_id: int, transcript: str | None) -> None:
    chunks = _split_document_into_chunks(transcript)
    if not chunks:
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM meeting_chunks WHERE meeting_id = $1", meeting_id)
        return

    texts = [str(chunk["content"]) for chunk in chunks]
    try:
        embeddings = embed_texts(texts)
    except Exception as exc:
        logger.warning("Embedding generation failed for meeting %s: %s", meeting_id, exc)
        embeddings = [None] * len(chunks)

    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute("DELETE FROM meeting_chunks WHERE meeting_id = $1", meeting_id)
            for chunk, emb in zip(chunks, embeddings):
                await conn.execute(
                    """
                    INSERT INTO meeting_chunks (
                        workspace_id, meeting_id, chunk_index, char_start, char_end, content, search_vector, embedding
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, to_tsvector('english', $6), $7)
                    ON CONFLICT (meeting_id, chunk_index) DO UPDATE
                    SET workspace_id = EXCLUDED.workspace_id,
                        char_start = EXCLUDED.char_start,
                        char_end = EXCLUDED.char_end,
                        content = EXCLUDED.content,
                        search_vector = EXCLUDED.search_vector,
                        embedding = EXCLUDED.embedding
                    """,
                    workspace_id,
                    meeting_id,
                    int(chunk["chunk_index"]),
                    int(chunk["char_start"]),
                    int(chunk["char_end"]),
                    str(chunk["content"]),
                    str(emb) if emb else None,
                )
    logger.info("Chunked+embedded meeting %s: %d chunks", meeting_id, len(chunks))


async def ensure_meeting_chunks(pool, workspace_id: int, meeting_ids: list[int]) -> None:
    target_ids = [int(item) for item in meeting_ids if item]
    if not target_ids:
        return
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT m.id, m.transcript, m.workspace_id
            FROM meetings m
            WHERE m.workspace_id = $1
              AND m.id = ANY($2::int[])
              AND COALESCE(m.transcript, '') <> ''
              AND NOT EXISTS (
                    SELECT 1
                    FROM meeting_chunks mc
                    WHERE mc.meeting_id = m.id
                )
            """,
            workspace_id,
            target_ids,
        )
    for row in rows:
        try:
            await replace_meeting_chunks(pool, row["id"], row["workspace_id"], row["transcript"])
        except Exception as exc:
            logger.warning("Meeting chunk backfill failed for meeting %s: %s", row["id"], exc)


async def backfill_meeting_chunks(pool) -> None:
    """Chunk all existing meetings that have transcripts but no chunks yet."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT m.id, m.workspace_id, m.transcript
            FROM meetings m
            WHERE COALESCE(m.transcript, '') <> ''
              AND m.workspace_id IS NOT NULL
              AND NOT EXISTS (
                    SELECT 1 FROM meeting_chunks mc WHERE mc.meeting_id = m.id
                )
            """
        )
    if not rows:
        return
    logger.info("Backfilling meeting chunks: %d meetings to process", len(rows))
    for row in rows:
        try:
            await replace_meeting_chunks(pool, row["id"], row["workspace_id"], row["transcript"])
        except Exception as exc:
            logger.warning("Meeting chunk backfill failed for meeting %s: %s", row["id"], exc)
    logger.info("Meeting chunk backfill complete")


# ---------------------------------------------------------------------------
# Query / retrieval helpers
# ---------------------------------------------------------------------------


def _document_query_text(*parts: Any) -> str:
    stopwords = {
        "the", "and", "for", "with", "that", "this", "from", "into", "your", "their",
        "what", "when", "where", "which", "will", "would", "could", "should", "have",
        "has", "had", "about", "using", "used", "need", "needs", "task", "deliverable",
    }
    terms: list[str] = []
    seen: set[str] = set()
    for part in parts:
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9._/-]{1,}", str(part or "")):
            normalized = token.strip("._/-").lower()
            if len(normalized) < 3 or normalized in stopwords or normalized in seen:
                continue
            seen.add(normalized)
            terms.append(normalized)
            if len(terms) >= DOCUMENT_QUERY_TERM_LIMIT:
                return " ".join(terms)
    return " ".join(terms)


def _serialize_document_ref(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "document_id": int(item.get("document_id") or 0),
        "filename": str(item.get("filename") or "").strip(),
        "chunk_index": int(item.get("chunk_index") or 0),
        "snippet": _excerpt_text(item.get("snippet") or item.get("content"), 420),
        "score": round(float(item.get("score") or 0.0), 4),
    }


def _dedupe_document_refs(items: list[dict[str, Any]], *, limit: int = DOCUMENT_RETRIEVAL_LIMIT) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for item in items:
        key = (int(item.get("document_id") or 0), int(item.get("chunk_index") or 0))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(_serialize_document_ref(item))
        if len(deduped) >= limit:
            break
    return deduped


async def retrieve_document_evidence(
    pool,
    workspace_id: int,
    document_ids: list[int],
    query_text: str,
    *,
    limit: int = DOCUMENT_RETRIEVAL_LIMIT,
) -> list[dict[str, Any]]:
    doc_ids = [int(item) for item in document_ids if item]
    if not doc_ids:
        return []
    await ensure_document_chunks(pool, workspace_id, doc_ids)
    search_query = _document_query_text(query_text)
    async with pool.acquire() as conn:
        rows: list[Any] = []
        has_embeddings = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM document_chunks WHERE document_id = ANY($1::int[]) AND embedding IS NOT NULL LIMIT 1)",
            doc_ids,
        )
        if has_embeddings and query_text and query_text.strip():
            try:
                query_embedding = embed_texts([query_text])[0]
                rows = await conn.fetch(
                    """
                    SELECT dc.document_id, d.filename, dc.chunk_index, dc.char_start, dc.char_end, dc.content,
                           1 - (dc.embedding <=> $3) AS score
                    FROM document_chunks dc
                    JOIN documents d ON d.id = dc.document_id
                    WHERE dc.workspace_id = $1
                      AND dc.document_id = ANY($2::int[])
                      AND dc.embedding IS NOT NULL
                    ORDER BY dc.embedding <=> $3
                    LIMIT $4
                    """,
                    workspace_id,
                    doc_ids,
                    str(query_embedding),
                    max(limit * 3, limit),
                )
                logger.info("RAG document retrieval: %d chunks for %d documents", len(rows), len(doc_ids))
            except Exception as exc:
                logger.warning("RAG document retrieval failed, falling back to FTS: %s", exc)
                rows = []
        if not rows and search_query:
            rows = await conn.fetch(
                """
                SELECT dc.document_id, d.filename, dc.chunk_index, dc.char_start, dc.char_end, dc.content,
                       ts_rank_cd(dc.search_vector, plainto_tsquery('english', $3)) AS score
                FROM document_chunks dc
                JOIN documents d ON d.id = dc.document_id
                WHERE dc.workspace_id = $1
                  AND dc.document_id = ANY($2::int[])
                  AND dc.search_vector @@ plainto_tsquery('english', $3)
                ORDER BY score DESC, dc.document_id, dc.chunk_index
                LIMIT $4
                """,
                workspace_id,
                doc_ids,
                search_query,
                max(limit * 3, limit),
            )
        if rows:
            results: list[dict[str, Any]] = []
            per_doc_counts: dict[int, int] = {}
            for row in rows:
                document_id = int(row["document_id"])
                if per_doc_counts.get(document_id, 0) >= 2:
                    continue
                per_doc_counts[document_id] = per_doc_counts.get(document_id, 0) + 1
                results.append({
                    "document_id": document_id,
                    "filename": row["filename"],
                    "chunk_index": int(row["chunk_index"]),
                    "char_start": int(row["char_start"]),
                    "char_end": int(row["char_end"]),
                    "content": row["content"],
                    "snippet": _excerpt_text(row["content"], 420),
                    "score": float(row["score"] or 0.0),
                })
                if len(results) >= limit:
                    break
            if results:
                return _dedupe_document_refs(results, limit=limit)
        fallback_rows = await conn.fetch(
            """
            SELECT id, filename, executive_summary, extracted_text, mime_type
            FROM documents
            WHERE workspace_id = $1 AND id = ANY($2::int[])
            ORDER BY uploaded_at DESC
            LIMIT $3
            """,
            workspace_id,
            doc_ids,
            min(limit, 3),
        )
    fallback: list[dict[str, Any]] = []
    for row in fallback_rows:
        snippet = _excerpt_text(row["executive_summary"] or row["extracted_text"], 420)
        if snippet:
            fallback.append({
                "document_id": int(row["id"]),
                "filename": row["filename"],
                "chunk_index": 0,
                "snippet": snippet,
                "score": 0.0,
            })
        else:
            fallback.append({
                "document_id": int(row["id"]),
                "filename": row["filename"],
                "chunk_index": 0,
                "snippet": f'Asset type: {row["mime_type"] or "unknown"} (no extracted text available).',
                "score": 0.0,
            })
    return _dedupe_document_refs(fallback, limit=limit)


async def retrieve_meeting_evidence(
    pool,
    workspace_id: int,
    meeting_ids: list[int],
    query_text: str,
    *,
    limit: int = DOCUMENT_RETRIEVAL_LIMIT,
) -> list[dict[str, Any]]:
    mid_list = [int(item) for item in meeting_ids if item]
    if not mid_list:
        return []
    await ensure_meeting_chunks(pool, workspace_id, mid_list)
    search_query = _document_query_text(query_text)
    async with pool.acquire() as conn:
        rows: list[Any] = []
        has_embeddings = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM meeting_chunks WHERE meeting_id = ANY($1::int[]) AND embedding IS NOT NULL LIMIT 1)",
            mid_list,
        )
        if has_embeddings and query_text and query_text.strip():
            try:
                query_embedding = embed_texts([query_text])[0]
                rows = await conn.fetch(
                    """
                    SELECT mc.meeting_id, m.title AS meeting_title, mc.chunk_index,
                           mc.char_start, mc.char_end, mc.content,
                           1 - (mc.embedding <=> $3) AS score
                    FROM meeting_chunks mc
                    JOIN meetings m ON m.id = mc.meeting_id
                    WHERE mc.workspace_id = $1
                      AND mc.meeting_id = ANY($2::int[])
                      AND mc.embedding IS NOT NULL
                    ORDER BY mc.embedding <=> $3
                    LIMIT $4
                    """,
                    workspace_id,
                    mid_list,
                    str(query_embedding),
                    max(limit * 3, limit),
                )
                logger.info("RAG meeting retrieval: %d chunks for %d meetings", len(rows), len(mid_list))
            except Exception as exc:
                logger.warning("RAG meeting retrieval failed, falling back to FTS: %s", exc)
                rows = []
        if not rows and search_query:
            rows = await conn.fetch(
                """
                SELECT mc.meeting_id, m.title AS meeting_title, mc.chunk_index,
                       mc.char_start, mc.char_end, mc.content,
                       ts_rank_cd(mc.search_vector, plainto_tsquery('english', $3)) AS score
                FROM meeting_chunks mc
                JOIN meetings m ON m.id = mc.meeting_id
                WHERE mc.workspace_id = $1
                  AND mc.meeting_id = ANY($2::int[])
                  AND mc.search_vector @@ plainto_tsquery('english', $3)
                ORDER BY score DESC, mc.meeting_id, mc.chunk_index
                LIMIT $4
                """,
                workspace_id,
                mid_list,
                search_query,
                max(limit * 3, limit),
            )
        if rows:
            results: list[dict[str, Any]] = []
            per_meeting_counts: dict[int, int] = {}
            for row in rows:
                meeting_id = int(row["meeting_id"])
                if per_meeting_counts.get(meeting_id, 0) >= 2:
                    continue
                per_meeting_counts[meeting_id] = per_meeting_counts.get(meeting_id, 0) + 1
                results.append({
                    "meeting_id": meeting_id,
                    "meeting_title": row["meeting_title"] or f"Meeting {meeting_id}",
                    "chunk_index": int(row["chunk_index"]),
                    "char_start": int(row["char_start"]),
                    "char_end": int(row["char_end"]),
                    "content": row["content"],
                    "snippet": _excerpt_text(row["content"], 420),
                    "score": float(row["score"] or 0.0),
                })
                if len(results) >= limit:
                    break
            if results:
                return results
        fallback_rows = await conn.fetch(
            """
            SELECT id, title, summary, transcript
            FROM meetings
            WHERE workspace_id = $1 AND id = ANY($2::int[])
            ORDER BY date DESC
            LIMIT $3
            """,
            workspace_id,
            mid_list,
            min(limit, 3),
        )
    fallback: list[dict[str, Any]] = []
    for row in fallback_rows:
        snippet = _excerpt_text(row["summary"] or row["transcript"], 420)
        if snippet:
            fallback.append({
                "meeting_id": int(row["id"]),
                "meeting_title": row["title"] or f"Meeting {row['id']}",
                "chunk_index": 0,
                "snippet": snippet,
                "score": 0.0,
            })
    return fallback


def document_evidence_prompt_block(items: list[dict[str, Any]], *, header: str = "Document evidence from attached reference documents") -> str:
    evidence = _dedupe_document_refs(items, limit=DOCUMENT_RETRIEVAL_LIMIT)
    if not evidence:
        return ""
    lines = [f"{header}:"]
    for index, item in enumerate(evidence, start=1):
        filename = item.get("filename") or f"Document {item.get('document_id')}"
        lines.append(
            f'- [D{index}] {filename} '
            f'(chunk {int(item.get("chunk_index") or 0) + 1}): {item.get("snippet") or ""}'
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Document row serialisation
# ---------------------------------------------------------------------------


def serialize_document_row(
    row: Any,
    *,
    include_text: bool = False,
) -> dict[str, Any]:
    from document_processing import _supports_document_text_extraction  # noqa: PLC0415

    data = dict(row)
    for dt_field in ("uploaded_at", "analyzed_at", "created_at"):
        if isinstance(data.get(dt_field), datetime):
            data[dt_field] = data[dt_field].isoformat()
    extracted_text = data.get("extracted_text")
    mime_type = (data.get("mime_type") or "").lower()
    supports_text = _supports_document_text_extraction(data.get("filename") or "", mime_type)
    if isinstance(data.get("key_takeaways"), str):
        try:
            data["key_takeaways"] = json.loads(data["key_takeaways"])
        except Exception:
            data["key_takeaways"] = []
    if not isinstance(data.get("key_takeaways"), list):
        data["key_takeaways"] = []
    tables = data.get("tables_json")
    if isinstance(tables, str):
        try:
            tables = json.loads(tables)
        except Exception:
            tables = []
    if not isinstance(tables, list):
        tables = []
    data["tables_json"] = tables
    data["is_image"] = mime_type.startswith("image/")
    data["has_text"] = bool(str(extracted_text or "").strip())
    data["has_summary"] = bool(str(data.get("executive_summary") or "").strip())
    data["has_tables"] = bool(tables)
    data["has_preview"] = bool(data.get("preview_pdf_key"))
    data["pending"] = supports_text and extracted_text is None
    if data.get("canvas_file_id"):
        data["source"] = "canvas"
    elif data.get("drive_file_id"):
        data["source"] = "drive"
    else:
        data["source"] = "upload"
    if not include_text:
        data.pop("extracted_text", None)
        data.pop("tables_json", None)
    data.pop("preview_pdf_key", None)
    return data


# ---------------------------------------------------------------------------
# LLM-powered document analysis
# ---------------------------------------------------------------------------


async def summarize_document_text(pool, text: str, workspace_id: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Call the LLM runner to generate executive summary + key takeaways for a document."""
    # Import here to avoid circular imports at module load time
    from services.llm_prefs import get_workspace_llm_preferences, _resolve_task_llm  # noqa: PLC0415

    excerpt = (text or "").strip()
    if not excerpt:
        raise RuntimeError("Document text is empty.")
    excerpt = excerpt[:120_000]
    preferences = await get_workspace_llm_preferences(pool, workspace_id)
    provider, model = _resolve_task_llm(preferences, "generate")
    # Import LLM call helper lazily to avoid circular import
    from llm import call_llm_runner_json  # noqa: PLC0415

    payload, meta = await call_llm_runner_json(
        [{
            "role": "user",
            "content": (
                "Create an executive summary for the following document. "
                "Return ONLY valid JSON with keys \"executive_summary\" and \"key_takeaways\". "
                "\"executive_summary\" must be a concise high-level summary in 2-4 sentences. "
                "\"key_takeaways\" must be an array of 3-6 short bullet-ready strings.\n\n"
                f"<document>\n{excerpt}\n</document>"
            ),
        }],
        provider=provider,
        model=model,
        use_case="chat",
        max_tokens=1800,
    )
    from services.text_svc import _json_dict  # noqa: PLC0415

    payload = _json_dict(payload)
    takeaways = payload.get("key_takeaways") or []
    if isinstance(takeaways, str):
        takeaways = [takeaways]
    return {
        "executive_summary": str(payload.get("executive_summary") or "").strip(),
        "key_takeaways": [str(item).strip() for item in takeaways if str(item).strip()][:6],
    }, meta


async def store_document_analysis(pool, doc_id: int, summary_payload: dict[str, Any], meta: dict[str, Any]) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            """UPDATE documents
               SET executive_summary = $1,
                   key_takeaways = $2::jsonb,
                   analyzed_at = NOW(),
                   analysis_provider = $3,
                   analysis_model = $4
               WHERE id = $5""",
            summary_payload.get("executive_summary") or "",
            json.dumps(summary_payload.get("key_takeaways") or []),
            meta.get("provider"),
            meta.get("model"),
            doc_id,
        )


async def analyze_document_and_store(pool, doc_id: int, workspace_id: int, extracted_text: str) -> None:
    summary_payload, meta = await summarize_document_text(pool, extracted_text, workspace_id)
    await store_document_analysis(pool, doc_id, summary_payload, meta)


# ---------------------------------------------------------------------------
# OCR / conversion helpers
# ---------------------------------------------------------------------------


def extract_image_text_ocr(data: bytes) -> str | None:
    """Extract text from image bytes (any format including HEIC) using Tesseract OCR."""
    try:
        from PIL import Image, ImageOps  # noqa: PLC0415
        import pytesseract  # noqa: PLC0415

        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        text = pytesseract.image_to_string(img).strip()
        return text if text else None
    except Exception as e:
        logger.warning("Image OCR failed: %s", e)
        return None


_OFFICE_EXTS = {".docx", ".doc", ".pptx", ".xlsx", ".xls"}
_OFFICE_MIMES = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel",
}


def linearize_pdf_sync(data: bytes) -> bytes:
    """Return a linearized (Fast Web View) copy of a PDF."""
    try:
        import pikepdf  # noqa: PLC0415

        with pikepdf.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                rotate = int(page.get("/Rotate", 0)) % 360
                if rotate != 0:
                    page.rotate(rotate, relative=False)
            out = io.BytesIO()
            pdf.save(out, linearize=True)
            result = out.getvalue()
        return result if result else data
    except Exception as exc:
        logger.warning("PDF linearization skipped: %s", exc)
        return data


def rotate_pdf_pages_sync(
    data: bytes,
    degrees: int,
    landscape_only: bool = True,
    page_indices: list[int] | None = None,
) -> bytes:
    """Rotate pages in a PDF and return re-linearized bytes."""
    try:
        import pikepdf  # noqa: PLC0415

        with pikepdf.open(io.BytesIO(data)) as pdf:
            n = len(pdf.pages)
            targets = page_indices if page_indices is not None else list(range(n))
            for i in targets:
                if i < 0 or i >= n:
                    continue
                page = pdf.pages[i]
                if landscape_only:
                    mb = page.MediaBox
                    w = float(mb[2]) - float(mb[0])
                    h = float(mb[3]) - float(mb[1])
                    existing = int(page.get("/Rotate", 0)) % 360
                    if existing in (90, 270):
                        w, h = h, w
                    if w <= h:
                        continue
                page.rotate(degrees, relative=True)
            out = io.BytesIO()
            pdf.save(out, linearize=True)
            result = out.getvalue()
        return result if result else data
    except Exception as exc:
        logger.warning("PDF rotation failed: %s", exc)
        return data


def convert_to_pdf_sync(data: bytes, filename: str) -> bytes | None:
    """Convert an Office document to PDF using LibreOffice. Returns PDF bytes or None."""
    ext = os.path.splitext(filename.lower())[1] or ".bin"
    with tempfile.TemporaryDirectory(prefix="lo-convert-") as tmpdir:
        src = os.path.join(tmpdir, f"input{ext}")
        with open(src, "wb") as f:
            f.write(data)
        env = os.environ.copy()
        env["HOME"] = tmpdir
        result = subprocess.run(
            [
                "soffice", "--headless", "--norestore", "--nofirststartwizard",
                "--convert-to", "pdf", "--outdir", tmpdir, src,
            ],
            capture_output=True,
            timeout=120,
            env=env,
        )
        if result.returncode != 0:
            logger.warning("soffice failed for %s: %s", filename, result.stderr.decode()[-500:])
            return None
        pdf_path = os.path.join(tmpdir, os.path.splitext("input" + ext)[0] + ".pdf")
        if not os.path.exists(pdf_path):
            logger.warning("soffice ran but no PDF output found for %s", filename)
            return None
        with open(pdf_path, "rb") as f:
            return f.read()


# ---------------------------------------------------------------------------
# Extract-and-store background task
# ---------------------------------------------------------------------------


async def extract_and_store(pool, minio_session, doc_id: int, workspace_id: int, data: bytes, mime_type: str, filename: str) -> None:
    """Background task: extract text and update documents row."""
    from document_processing import (  # noqa: PLC0415
        _extract_tables_sync,
        _extract_text_sync,
        _extract_text_pptx_hybrid_sync,
    )
    from services.storage import get_minio_client  # noqa: PLC0415

    try:
        is_image = mime_type and mime_type.startswith("image/")
        is_pdf = (mime_type or "").lower() == "application/pdf" or filename.lower().endswith(".pdf")

        if is_pdf and not is_image:
            try:
                lin_data = await asyncio.to_thread(linearize_pdf_sync, data)
                if lin_data and len(lin_data) > 0:
                    async with pool.acquire() as conn:
                        key_row = await conn.fetchrow("SELECT object_key FROM documents WHERE id = $1", doc_id)
                    if key_row and key_row["object_key"]:
                        async with get_minio_client(minio_session) as client:
                            await client.put_object(
                                Bucket=MINIO_BUCKET, Key=key_row["object_key"],
                                Body=lin_data, ContentType="application/pdf",
                            )
                        logger.info("Linearized uploaded PDF for doc %d (%d -> %d bytes)", doc_id, len(data), len(lin_data))
                        data = lin_data
            except Exception as exc:
                logger.warning("PDF linearization step failed for doc %d: %s", doc_id, exc)

        ext = os.path.splitext(filename.lower())[1]
        preview_pdf_bytes: bytes | None = None
        if not is_image and (ext in _OFFICE_EXTS or (mime_type or "").lower() in _OFFICE_MIMES):
            try:
                raw_pdf = await asyncio.to_thread(convert_to_pdf_sync, data, filename)
                if raw_pdf:
                    preview_pdf_bytes = raw_pdf
                    lin_pdf = await asyncio.to_thread(linearize_pdf_sync, raw_pdf)
                    pdf_key = f"workspaces/{workspace_id}/previews/{uuid.uuid4()}_{os.path.splitext(filename)[0]}.pdf"
                    async with get_minio_client(minio_session) as client:
                        await client.put_object(
                            Bucket=MINIO_BUCKET, Key=pdf_key,
                            Body=lin_pdf, ContentType="application/pdf",
                        )
                    async with pool.acquire() as conn:
                        await conn.execute(
                            "UPDATE documents SET preview_pdf_key = $1 WHERE id = $2",
                            pdf_key, doc_id,
                        )
                    logger.info("PDF preview generated for doc %d", doc_id)
            except Exception as exc:
                logger.error("PDF conversion failed for doc %d: %s", doc_id, exc)

        is_pptx = (ext == ".pptx" or (mime_type or "").lower() == "application/vnd.openxmlformats-officedocument.presentationml.presentation")
        if is_image:
            extracted = await asyncio.to_thread(extract_image_text_ocr, data)
            tables = []
        elif is_pptx:
            extracted = await asyncio.to_thread(_extract_text_pptx_hybrid_sync, data, filename, preview_pdf_bytes)
            tables = await asyncio.to_thread(_extract_tables_sync, data, mime_type, filename)
        else:
            extracted = await asyncio.to_thread(_extract_text_sync, data, mime_type, filename)
            tables = await asyncio.to_thread(_extract_tables_sync, data, mime_type, filename)
        if extracted:
            extracted = extracted.replace("\x00", "")
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE documents SET extracted_text = $1, tables_json = $2::jsonb WHERE id = $3",
                extracted,
                json.dumps(tables),
                doc_id,
            )
        await replace_document_chunks(pool, doc_id, workspace_id, extracted)
        logger.info("Text extraction complete for doc %d: %s chars", doc_id, len(extracted) if extracted else 0)
        if extracted and extracted.strip():
            try:
                await analyze_document_and_store(pool, doc_id, workspace_id, extracted)
                logger.info("Document summary complete for doc %d", doc_id)
            except Exception as exc:
                logger.error("Document analysis error for doc %d: %s", doc_id, exc)
    except Exception as e:
        logger.error("Text extraction error for doc %d: %s", doc_id, e)


# ---------------------------------------------------------------------------
# Content-Disposition helper
# ---------------------------------------------------------------------------


def content_disposition(disposition: str, filename: str) -> str:
    """Build a Content-Disposition header value that handles non-ASCII filenames."""
    try:
        filename.encode("latin-1")
        return f'{disposition}; filename="{filename}"'
    except (UnicodeEncodeError, UnicodeDecodeError):
        from urllib.parse import quote as _quote  # noqa: PLC0415

        return f"{disposition}; filename*=UTF-8''{_quote(filename, safe='')}"


# ---------------------------------------------------------------------------
# Store generated documents (from generate task output)
# ---------------------------------------------------------------------------


async def store_generated_document(
    pool,
    minio_session,
    workspace_id: int,
    filename: str,
    mime_type: str,
    data: bytes,
    extracted_text: str,
) -> dict[str, Any]:
    from services.storage import get_minio_client  # noqa: PLC0415

    object_key = f"workspaces/{workspace_id}/documents/{uuid.uuid4()}_{filename}"
    async with get_minio_client(minio_session) as client:
        await client.put_object(
            Bucket=MINIO_BUCKET,
            Key=object_key,
            Body=data,
            ContentType=mime_type,
        )
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """INSERT INTO documents (
                   workspace_id, filename, object_key, file_size, mime_type, extracted_text,
                   executive_summary, key_takeaways, analyzed_at
               )
               VALUES ($1, $2, $3, $4, $5, $6, NULL, '[]'::jsonb, NULL)
               RETURNING id, filename, file_size, mime_type, uploaded_at, extracted_text,
                         executive_summary, key_takeaways, analyzed_at""",
            workspace_id,
            filename,
            object_key,
            len(data),
            mime_type,
            extracted_text,
        )
    await replace_document_chunks(pool, row["id"], workspace_id, extracted_text)
    return serialize_document_row(row)
