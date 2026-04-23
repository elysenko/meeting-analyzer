#!/usr/bin/env python3
"""Worker: process exactly ONE document by ID. Called by backfill_coordinator.py."""
import asyncio, gc, io, json, logging, os, subprocess, sys, tempfile, uuid
import asyncpg, aiobotocore.session
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    stream=sys.stdout, force=True)
log = logging.getLogger("worker")

DB_URL   = "postgresql://analyzer:analyzer-pass-2026@analyzer-postgres:5432/analyzer"
ENDPOINT = f"http://{os.getenv('MINIO_ENDPOINT','minio.minio.svc.cluster.local:9000')}"
BUCKET   = os.getenv("MINIO_BUCKET","meeting-analyzer")
AK       = os.getenv("MINIO_ACCESS_KEY","2MyaV3Y29l7zcIpS")
SK       = os.getenv("MINIO_SECRET_KEY","CxKyR5waqrMqVa5Pb7o/YqwVUAmcTROW0Scpjk5l9Qg=")

sys.path.insert(0, "/app")
from document_processing import _extract_text_sync, _extract_text_pptx_hybrid_sync, _extract_tables_sync


def _convert_to_pdf(data, filename):
    ext = os.path.splitext(filename.lower())[1] or ".bin"
    with tempfile.TemporaryDirectory(prefix="lo-") as d:
        src = os.path.join(d, f"input{ext}")
        open(src,"wb").write(data)
        env = {**os.environ,"HOME":d}
        r = subprocess.run(["soffice","--headless","--norestore","--nofirststartwizard",
                            "--convert-to","pdf","--outdir",d,src],
                           capture_output=True, timeout=120, env=env)
        if r.returncode != 0:
            log.warning("soffice failed: %s", r.stderr.decode()[-200:])
            return None
        p = os.path.join(d,"input.pdf")
        return open(p,"rb").read() if os.path.exists(p) else None


def _linearize(data):
    try:
        import pikepdf
        with pikepdf.open(io.BytesIO(data)) as pdf:
            out = io.BytesIO(); pdf.save(out, linearize=True)
            return out.getvalue() or data
    except: return data


async def main(doc_id: int):
    conn = await asyncpg.connect(DB_URL)
    row = await conn.fetchrow(
        "SELECT id,filename,mime_type,object_key,workspace_id,preview_pdf_key FROM documents WHERE id=$1",
        doc_id)
    if not row:
        log.error("Doc %d not found", doc_id); await conn.close(); return

    fname = row["filename"]; mime = row["mime_type"] or ""
    okey = row["object_key"]; ws = row["workspace_id"]
    has_prev = row["preview_pdf_key"] is not None
    ext = os.path.splitext(fname.lower())[1]
    log.info("Doc %d: %s", doc_id, fname)

    session = aiobotocore.session.get_session()
    async with session.create_client("s3", endpoint_url=ENDPOINT,
            aws_access_key_id=AK, aws_secret_access_key=SK, region_name="us-east-1") as s3:
        try:
            resp = await s3.get_object(Bucket=BUCKET, Key=okey)
            data = await resp["Body"].read()
        except Exception as e:
            log.error("  MinIO fetch failed: %s", e); await conn.close(); return

        pdf_bytes = None
        is_office = ext in {".docx",".doc",".pptx",".xlsx",".xls"}

        if is_office and not has_prev:
            log.info("  → generating preview")
            raw = await asyncio.to_thread(_convert_to_pdf, data, fname)
            if raw:
                pdf_bytes = raw
                lin = await asyncio.to_thread(_linearize, raw)
                key = f"workspaces/{ws}/previews/{uuid.uuid4()}_{os.path.splitext(fname)[0]}.pdf"
                await s3.put_object(Bucket=BUCKET, Key=key, Body=lin, ContentType="application/pdf")
                await conn.execute("UPDATE documents SET preview_pdf_key=$1 WHERE id=$2", key, doc_id)
                log.info("  preview stored")
        elif ext == ".pptx" and has_prev and row["preview_pdf_key"]:
            try:
                r2 = await s3.get_object(Bucket=BUCKET, Key=row["preview_pdf_key"])
                pdf_bytes = await r2["Body"].read()
            except: pass

    is_pptx = ext == ".pptx" or "presentationml" in mime
    if is_pptx:
        log.info("  → PPTX hybrid OCR")
        extracted = await asyncio.to_thread(_extract_text_pptx_hybrid_sync, data, fname, pdf_bytes)
    else:
        extracted = await asyncio.to_thread(_extract_text_sync, data, mime, fname)

    if extracted:
        extracted = extracted.replace("\x00","")
    tables = await asyncio.to_thread(_extract_tables_sync, data, mime, fname)
    del data, pdf_bytes; gc.collect()

    await conn.execute(
        "UPDATE documents SET extracted_text=$1, tables_json=$2::jsonb WHERE id=$3",
        extracted, json.dumps(tables), doc_id)
    log.info("  text: %d chars", len(extracted) if extracted else 0)
    await conn.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: backfill_worker.py <doc_id>"); sys.exit(1)
    asyncio.run(main(int(sys.argv[1])))
