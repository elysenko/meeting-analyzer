#!/usr/bin/env python3
"""
Coordinator: runs backfill_worker.py for each document in its own subprocess,
then runs embedding for all updated docs.
Each subprocess exits cleanly, guaranteeing full memory release between docs.
"""
import asyncio, logging, os, re, subprocess, sys
import asyncpg
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    stream=sys.stdout, force=True)
log = logging.getLogger("coord")

DB_URL        = "postgresql://analyzer:analyzer-pass-2026@analyzer-postgres:5432/analyzer"
WORKER        = "/tmp/backfill_worker.py"
CHUNK_TARGET  = 1400
CHUNK_MIN     = 700
CHUNK_OVERLAP = 220


def _normalize(text):
    c = (text or "").replace("\r\n","\n").replace("\r","\n")
    c = re.sub(r"[ \t]+\n","\n",c)
    c = re.sub(r"\n{3,}","\n\n",c)
    c = re.sub(r"[ \t]{2,}"," ",c)
    return c.strip()


def _split(text):
    n = _normalize(text)
    if not n: return []
    chunks, start, L = [], 0, len(n)
    while start < L:
        hard = min(start + CHUNK_TARGET, L)
        end = hard
        if hard < L:
            cand = -1
            ss = min(start + CHUNK_MIN, hard)
            for m in ("\n\n",". ","? ","! ","; ",": "):
                f = n.rfind(m, ss, hard)
                if f >= 0: cand = max(cand, f + len(m))
            if cand > start: end = cand
        if end <= start: end = hard
        content = n[start:end].strip()
        if content:
            chunks.append({"chunk_index":len(chunks),"char_start":start,"char_end":end,"content":content})
        if end >= L: break
        start = max(end - CHUNK_OVERLAP, start + 1)
        while start < L and n[start].isspace(): start += 1
    return chunks


_model = None
def _embed(texts):
    global _model
    if _model is None:
        from fastembed import TextEmbedding
        _model = TextEmbedding("BAAI/bge-small-en-v1.5", cache_dir="/models/fastembed")
        log.info("Embedding model loaded")
    return [e.tolist() for e in _model.embed(texts)]


async def rechunk(conn, doc_id, ws, text):
    chunks = _split(text)
    await conn.execute("DELETE FROM document_chunks WHERE document_id=$1", doc_id)
    if not chunks: return 0
    embeddings = await asyncio.to_thread(_embed, [c["content"] for c in chunks])
    async with conn.transaction():
        for c, emb in zip(chunks, embeddings):
            await conn.execute("""
                INSERT INTO document_chunks
                  (workspace_id,document_id,chunk_index,char_start,char_end,content,search_vector,embedding)
                VALUES ($1,$2,$3,$4,$5,$6,to_tsvector('english',$6),$7)
                ON CONFLICT (document_id,chunk_index) DO UPDATE
                SET workspace_id=EXCLUDED.workspace_id,char_start=EXCLUDED.char_start,
                    char_end=EXCLUDED.char_end,content=EXCLUDED.content,
                    search_vector=EXCLUDED.search_vector,embedding=EXCLUDED.embedding
            """, ws, doc_id, int(c["chunk_index"]), int(c["char_start"]), int(c["char_end"]),
                str(c["content"]), str(emb) if emb else None)
    return len(chunks)


async def main():
    conn = await asyncpg.connect(DB_URL)

    # ── Phase 1: run worker subprocess per document ───────────────────────
    rows = await conn.fetch("""
        SELECT id FROM documents
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
    doc_ids = [r["id"] for r in rows]
    log.info("Phase 1: %d documents to process", len(doc_ids))

    for i, doc_id in enumerate(doc_ids, 1):
        log.info("[%d/%d] doc %d", i, len(doc_ids), doc_id)
        result = subprocess.run(
            ["python3", WORKER, str(doc_id)],
            timeout=900,
            capture_output=False,   # let stdout/stderr pass through
        )
        if result.returncode != 0:
            log.error("  worker exited %d for doc %d", result.returncode, doc_id)

    # ── Phase 2: embed all docs with text but no chunks ───────────────────
    to_embed = await conn.fetch("""
        SELECT d.id, d.workspace_id, d.extracted_text
        FROM documents d
        WHERE d.extracted_text IS NOT NULL AND d.extracted_text != ''
          AND NOT EXISTS (SELECT 1 FROM document_chunks dc WHERE dc.document_id=d.id)
        ORDER BY d.id
    """)
    log.info("Phase 2: %d documents need embedding", len(to_embed))
    for i, row in enumerate(to_embed, 1):
        log.info("[%d/%d] embedding doc %d", i, len(to_embed), row["id"])
        try:
            n = await rechunk(conn, row["id"], row["workspace_id"], row["extracted_text"])
            log.info("  → %d chunks", n)
        except Exception as e:
            log.error("  embed error: %s", e)

    await conn.close()
    log.info("All done.")

if __name__ == "__main__":
    asyncio.run(main())
