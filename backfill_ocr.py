#!/usr/bin/env python3
"""
Backfill script: regenerate PDF previews and re-extract text (with Tesseract OCR
for PPTX) for all documents that need it.

Run inside the pod:
  kubectl exec -n whisper <pod> -- python3 /app/backfill_ocr.py
"""
import asyncio
import gc
import io
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import uuid

import aiobotocore.session
import asyncpg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    stream=__import__("sys").stdout, force=True)
log = logging.getLogger("backfill")

# ── Config (matches pod env vars) ─────────────────────────────────────────────
DB_URL = "postgresql://analyzer:analyzer-pass-2026@analyzer-postgres:5432/analyzer"
MINIO_ENDPOINT = f"http://{os.getenv('MINIO_ENDPOINT', 'minio.minio.svc.cluster.local:9000')}"
MINIO_BUCKET   = os.getenv("MINIO_BUCKET", "meeting-analyzer")
MINIO_ACCESS   = os.getenv("MINIO_ACCESS_KEY", "2MyaV3Y29l7zcIpS")
MINIO_SECRET   = os.getenv("MINIO_SECRET_KEY", "CxKyR5waqrMqVa5Pb7o/YqwVUAmcTROW0Scpjk5l9Qg=")

CHUNK_TARGET  = 1400
CHUNK_MIN     = 700
CHUNK_OVERLAP = 220

# ── Import document helpers (safe — no FastAPI side effects) ──────────────────
sys.path.insert(0, "/app")
from document_processing import (
    _extract_text_sync,
    _extract_text_pptx_hybrid_sync,
    _extract_tables_sync,
)


# ── Inlined helpers (avoid importing main_live.py) ────────────────────────────

def _convert_to_pdf_sync(data: bytes, filename: str) -> bytes | None:
    ext = os.path.splitext(filename.lower())[1] or ".bin"
    with tempfile.TemporaryDirectory(prefix="lo-bf-") as tmpdir:
        src = os.path.join(tmpdir, f"input{ext}")
        with open(src, "wb") as f:
            f.write(data)
        env = os.environ.copy()
        env["HOME"] = tmpdir
        result = subprocess.run(
            ["soffice", "--headless", "--norestore", "--nofirststartwizard",
             "--convert-to", "pdf", "--outdir", tmpdir, src],
            capture_output=True, timeout=120, env=env,
        )
        if result.returncode != 0:
            log.warning("soffice failed for %s: %s", filename, result.stderr.decode()[-300:])
            return None
        pdf_path = os.path.join(tmpdir, "input.pdf")
        if not os.path.exists(pdf_path):
            log.warning("soffice ran but no PDF for %s", filename)
            return None
        with open(pdf_path, "rb") as f:
            return f.read()


def _linearize_pdf_sync(data: bytes) -> bytes:
    try:
        import pikepdf
        with pikepdf.open(io.BytesIO(data)) as pdf:
            out = io.BytesIO()
            pdf.save(out, linearize=True)
            return out.getvalue() or data
    except Exception as e:
        log.warning("Linearization skipped: %s", e)
        return data


def _normalize_text(text: str | None) -> str:
    cleaned = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def _split_chunks(text: str | None) -> list[dict]:
    normalized = _normalize_text(text)
    if not normalized:
        return []
    chunks, start, length = [], 0, len(normalized)
    while start < length:
        hard_end = min(start + CHUNK_TARGET, length)
        end = hard_end
        if hard_end < length:
            candidate = -1
            search_start = min(start + CHUNK_MIN, hard_end)
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
            chunks.append({"chunk_index": len(chunks), "char_start": start, "char_end": end, "content": content})
        if end >= length:
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
        while start < length and normalized[start].isspace():
            start += 1
    return chunks


_embed_model = None

def _embed(texts: list[str]) -> list[list[float] | None]:
    global _embed_model
    try:
        from fastembed import TextEmbedding
        if _embed_model is None:
            _embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5",
                                         cache_dir="/models/fastembed")
        return [e.tolist() for e in _embed_model.embed(texts)]
    except Exception as e:
        log.warning("Embedding failed: %s", e)
        return [None] * len(texts)


# ── Core processing ────────────────────────────────────────────────────────────

async def fetch_file(client, key: str) -> bytes | None:
    try:
        resp = await client.get_object(Bucket=MINIO_BUCKET, Key=key)
        return await resp["Body"].read()
    except Exception as e:
        log.warning("MinIO fetch failed for %s: %s", key, e)
        return None


async def store_preview(client, workspace_id: int, filename: str, pdf_bytes: bytes) -> str:
    stem = os.path.splitext(filename)[0]
    key = f"workspaces/{workspace_id}/previews/{uuid.uuid4()}_{stem}.pdf"
    await client.put_object(Bucket=MINIO_BUCKET, Key=key, Body=pdf_bytes, ContentType="application/pdf")
    return key


async def replace_chunks(conn, doc_id: int, workspace_id: int, extracted: str | None):
    chunks = _split_chunks(extracted)
    await conn.execute("DELETE FROM document_chunks WHERE document_id = $1", doc_id)
    if not chunks:
        return
    texts = [c["content"] for c in chunks]
    embeddings = await asyncio.to_thread(_embed, texts)
    async with conn.transaction():
        for chunk, emb in zip(chunks, embeddings):
            await conn.execute(
                """INSERT INTO document_chunks
                   (workspace_id, document_id, chunk_index, char_start, char_end, content, search_vector, embedding)
                   VALUES ($1,$2,$3,$4,$5,$6,to_tsvector('english',$6),$7)
                   ON CONFLICT (document_id, chunk_index) DO UPDATE
                   SET workspace_id=EXCLUDED.workspace_id, char_start=EXCLUDED.char_start,
                       char_end=EXCLUDED.char_end, content=EXCLUDED.content,
                       search_vector=EXCLUDED.search_vector, embedding=EXCLUDED.embedding""",
                workspace_id, doc_id, int(chunk["chunk_index"]),
                int(chunk["char_start"]), int(chunk["char_end"]),
                str(chunk["content"]), str(emb) if emb else None,
            )
    log.info("  → %d chunks embedded", len(chunks))


async def process_doc(conn, client, row: dict):
    doc_id      = row["id"]
    filename    = row["filename"]
    mime_type   = row["mime_type"] or ""
    object_key  = row["object_key"]
    workspace_id = row["workspace_id"]
    has_preview = row["preview_pdf_key"] is not None
    ext = os.path.splitext(filename.lower())[1]

    log.info("Doc %d: %s  [mime=%s, has_preview=%s]", doc_id, filename, mime_type, has_preview)

    data = await fetch_file(client, object_key)
    if not data:
        log.error("  SKIP — could not fetch from MinIO")
        return

    preview_pdf_bytes: bytes | None = None
    gc.collect()  # free memory before heavy processing

    # ── Generate missing PDF preview ─────────────────────────────────────────
    is_office = ext in {".docx", ".doc", ".pptx", ".xlsx", ".xls"}
    if is_office and not has_preview:
        log.info("  Generating PDF preview via LibreOffice...")
        raw_pdf = await asyncio.to_thread(_convert_to_pdf_sync, data, filename)
        if raw_pdf:
            preview_pdf_bytes = raw_pdf
            lin_pdf = await asyncio.to_thread(_linearize_pdf_sync, raw_pdf)
            pdf_key = await store_preview(client, workspace_id, filename, lin_pdf)
            await conn.execute("UPDATE documents SET preview_pdf_key=$1 WHERE id=$2", pdf_key, doc_id)
            log.info("  Preview stored: %s", pdf_key)
        else:
            log.warning("  PDF conversion failed — preview skipped")
    elif ext == ".pptx" and has_preview and row["preview_pdf_key"]:
        # Reuse existing preview PDF for OCR
        preview_pdf_bytes = await fetch_file(client, row["preview_pdf_key"])

    # ── Extract / re-extract text ─────────────────────────────────────────────
    is_pptx = ext == ".pptx" or "presentationml.presentation" in mime_type
    if is_pptx:
        log.info("  Running hybrid PPTX OCR...")
        extracted = await asyncio.to_thread(
            _extract_text_pptx_hybrid_sync, data, filename, preview_pdf_bytes
        )
    else:
        extracted = await asyncio.to_thread(_extract_text_sync, data, mime_type, filename)

    if extracted:
        extracted = extracted.replace("\x00", "")

    tables = await asyncio.to_thread(_extract_tables_sync, data, mime_type, filename)

    await conn.execute(
        "UPDATE documents SET extracted_text=$1, tables_json=$2::jsonb WHERE id=$3",
        extracted, json.dumps(tables), doc_id,
    )
    log.info("  Text updated: %d chars", len(extracted) if extracted else 0)

    # ── Re-embed ──────────────────────────────────────────────────────────────
    await replace_chunks(conn, doc_id, workspace_id, extracted)
    data = None
    preview_pdf_bytes = None
    gc.collect()


async def main():
    conn = await asyncpg.connect(DB_URL)
    session = aiobotocore.session.get_session()

    async with session.create_client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS,
        aws_secret_access_key=MINIO_SECRET,
        region_name="us-east-1",
    ) as client:
        rows = await conn.fetch("""
            SELECT id, filename, mime_type, object_key, workspace_id, preview_pdf_key, extracted_text
            FROM documents
            WHERE
              -- PPTX: always re-extract with OCR
              mime_type = 'application/vnd.openxmlformats-officedocument.presentationml.presentation'
              OR filename ILIKE '%.pptx'
              -- Office docs missing a preview
              OR (
                (mime_type IN (
                  'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                  'application/msword',
                  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                  'application/vnd.ms-excel'
                ) OR filename ILIKE '%.docx' OR filename ILIKE '%.doc'
                  OR filename ILIKE '%.xlsx' OR filename ILIKE '%.xls')
                AND preview_pdf_key IS NULL
              )
              -- DOC files with no extracted text
              OR (
                (mime_type = 'application/msword' OR filename ILIKE '%.doc')
                AND (extracted_text IS NULL OR extracted_text = '')
              )
              -- PDFs with no extracted text
              OR (
                (mime_type = 'application/pdf' OR filename ILIKE '%.pdf')
                AND (extracted_text IS NULL OR extracted_text = '')
              )
            ORDER BY id
        """)

        log.info("Found %d documents to process", len(rows))
        for i, row in enumerate(rows, 1):
            log.info("[%d/%d]", i, len(rows))
            try:
                await process_doc(conn, client, dict(row))
            except Exception as e:
                log.error("  ERROR on doc %d: %s", row["id"], e)

    await conn.close()
    log.info("Backfill complete.")


if __name__ == "__main__":
    asyncio.run(main())
