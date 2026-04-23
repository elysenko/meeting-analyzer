#!/usr/bin/env python3
"""
Backfill Phase 1: Generate missing PDF previews + extract text.
NO embedding — keeps memory low. Run backfill_phase2.py afterwards.
"""
import asyncio, gc, io, json, logging, os, re, subprocess, sys, tempfile, uuid
import asyncpg, aiobotocore.session
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    stream=sys.stdout, force=True)
log = logging.getLogger("bf1")

DB_URL    = "postgresql://analyzer:analyzer-pass-2026@analyzer-postgres:5432/analyzer"
ENDPOINT  = f"http://{os.getenv('MINIO_ENDPOINT','minio.minio.svc.cluster.local:9000')}"
BUCKET    = os.getenv("MINIO_BUCKET", "meeting-analyzer")
AK        = os.getenv("MINIO_ACCESS_KEY", "2MyaV3Y29l7zcIpS")
SK        = os.getenv("MINIO_SECRET_KEY", "CxKyR5waqrMqVa5Pb7o/YqwVUAmcTROW0Scpjk5l9Qg=")

sys.path.insert(0, "/app")
from document_processing import _extract_text_sync, _extract_text_pptx_hybrid_sync, _extract_tables_sync


def _convert_to_pdf(data: bytes, filename: str) -> bytes | None:
    ext = os.path.splitext(filename.lower())[1] or ".bin"
    with tempfile.TemporaryDirectory(prefix="lo-") as d:
        src = os.path.join(d, f"input{ext}")
        open(src, "wb").write(data)
        env = {**os.environ, "HOME": d}
        r = subprocess.run(["soffice","--headless","--norestore","--nofirststartwizard",
                            "--convert-to","pdf","--outdir",d,src],
                           capture_output=True, timeout=120, env=env)
        if r.returncode != 0:
            log.warning("soffice failed: %s", r.stderr.decode()[-200:])
            return None
        p = os.path.join(d, "input.pdf")
        return open(p,"rb").read() if os.path.exists(p) else None


def _linearize(data: bytes) -> bytes:
    try:
        import pikepdf
        with pikepdf.open(io.BytesIO(data)) as pdf:
            out = io.BytesIO(); pdf.save(out, linearize=True)
            return out.getvalue() or data
    except: return data


async def fetch(client, key):
    try:
        r = await client.get_object(Bucket=BUCKET, Key=key)
        return await r["Body"].read()
    except Exception as e:
        log.warning("fetch %s: %s", key, e); return None


async def store_preview(client, ws, name, pdf):
    key = f"workspaces/{ws}/previews/{uuid.uuid4()}_{os.path.splitext(name)[0]}.pdf"
    await client.put_object(Bucket=BUCKET, Key=key, Body=pdf, ContentType="application/pdf")
    return key


async def process(conn, client, row):
    doc_id, fname, mime, okey, ws, has_prev = (
        row["id"], row["filename"], row["mime_type"] or "",
        row["object_key"], row["workspace_id"], row["preview_pdf_key"] is not None)
    ext = os.path.splitext(fname.lower())[1]
    log.info("Doc %d: %s", doc_id, fname)

    data = await fetch(client, okey)
    if not data:
        log.error("  SKIP — MinIO fetch failed"); return

    pdf_bytes = None
    is_office = ext in {".docx",".doc",".pptx",".xlsx",".xls"}

    # Generate missing preview
    if is_office and not has_prev:
        log.info("  → generating preview")
        raw = await asyncio.to_thread(_convert_to_pdf, data, fname)
        if raw:
            pdf_bytes = raw
            lin = await asyncio.to_thread(_linearize, raw)
            key = await store_preview(client, ws, fname, lin)
            await conn.execute("UPDATE documents SET preview_pdf_key=$1 WHERE id=$2", key, doc_id)
            log.info("  preview: %s", key)
    elif ext == ".pptx" and has_prev and row["preview_pdf_key"]:
        pdf_bytes = await fetch(client, row["preview_pdf_key"])

    # Extract text
    is_pptx = ext == ".pptx" or "presentationml" in mime
    if is_pptx:
        log.info("  → PPTX hybrid OCR")
        extracted = await asyncio.to_thread(_extract_text_pptx_hybrid_sync, data, fname, pdf_bytes)
    else:
        extracted = await asyncio.to_thread(_extract_text_sync, data, mime, fname)

    if extracted:
        extracted = extracted.replace("\x00", "")
    tables = await asyncio.to_thread(_extract_tables_sync, data, mime, fname)

    await conn.execute(
        "UPDATE documents SET extracted_text=$1, tables_json=$2::jsonb WHERE id=$3",
        extracted, json.dumps(tables), doc_id)
    log.info("  text: %d chars", len(extracted) if extracted else 0)

    data = pdf_bytes = None; gc.collect()


async def main():
    conn = await asyncpg.connect(DB_URL)
    session = aiobotocore.session.get_session()
    async with session.create_client("s3", endpoint_url=ENDPOINT,
            aws_access_key_id=AK, aws_secret_access_key=SK,
            region_name="us-east-1") as client:
        rows = await conn.fetch("""
            SELECT id, filename, mime_type, object_key, workspace_id, preview_pdf_key
            FROM documents
            WHERE
              mime_type='application/vnd.openxmlformats-officedocument.presentationml.presentation'
              OR filename ILIKE '%.pptx'
              OR ((mime_type IN (
                    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                    'application/msword',
                    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    'application/vnd.ms-excel')
                  OR filename ILIKE '%.docx' OR filename ILIKE '%.doc'
                  OR filename ILIKE '%.xlsx' OR filename ILIKE '%.xls')
                AND preview_pdf_key IS NULL)
              OR ((mime_type='application/msword' OR filename ILIKE '%.doc')
                AND (extracted_text IS NULL OR extracted_text=''))
              OR ((mime_type='application/pdf' OR filename ILIKE '%.pdf')
                AND (extracted_text IS NULL OR extracted_text=''))
            ORDER BY id
        """)
        log.info("Phase 1: %d documents", len(rows))
        for i, row in enumerate(rows, 1):
            log.info("[%d/%d]", i, len(rows))
            try:
                await process(conn, client, dict(row))
            except Exception as e:
                log.error("  ERROR doc %d: %s", row["id"], e)
    await conn.close()
    log.info("Phase 1 complete.")

if __name__ == "__main__":
    asyncio.run(main())
