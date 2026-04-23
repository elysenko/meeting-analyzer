#!/usr/bin/env python3
"""Backfill embeddings for all existing meeting and document chunks.

Usage:
  python3 backfill_embeddings.py [--batch-size 50] [--meetings-only] [--documents-only]

Connects to the same DATABASE_URL as the main app. Generates embeddings
for all chunks that have NULL embedding column using fastembed (ONNX-based,
no PyTorch required).
"""

import argparse
import asyncio
import logging
import os
import time

import asyncpg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("backfill-embeddings")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://analyzer:analyzer-pass-2026@analyzer-postgres:5432/analyzer",
)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")


def load_model():
    from fastembed import TextEmbedding
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = TextEmbedding(model_name=EMBEDDING_MODEL)
    logger.info("Model loaded")
    return model


def embed_batch(model, texts: list[str]) -> list[list[float]]:
    embeddings = list(model.embed(texts))
    return [e.tolist() for e in embeddings]


async def backfill_meeting_chunks(pool: asyncpg.Pool, model, batch_size: int):
    async with pool.acquire() as conn:
        total = await conn.fetchval(
            "SELECT count(*) FROM meeting_chunks WHERE embedding IS NULL AND content IS NOT NULL AND content != ''"
        )
    logger.info(f"Meeting chunks to embed: {total}")

    if total == 0:
        return

    processed = 0
    while True:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, content FROM meeting_chunks "
                "WHERE embedding IS NULL AND content IS NOT NULL AND content != '' "
                "ORDER BY id LIMIT $1",
                batch_size,
            )

        if not rows:
            break

        texts = [row["content"] for row in rows]
        ids = [row["id"] for row in rows]

        start = time.time()
        embeddings = embed_batch(model, texts)
        elapsed = time.time() - start

        async with pool.acquire() as conn:
            for chunk_id, emb in zip(ids, embeddings):
                await conn.execute(
                    "UPDATE meeting_chunks SET embedding = $1 WHERE id = $2",
                    str(emb), chunk_id,
                )

        processed += len(rows)
        logger.info(f"Meeting chunks: {processed}/{total} ({elapsed:.1f}s for {len(rows)} chunks)")


async def backfill_document_chunks(pool: asyncpg.Pool, model, batch_size: int):
    async with pool.acquire() as conn:
        total = await conn.fetchval(
            "SELECT count(*) FROM document_chunks WHERE embedding IS NULL AND content IS NOT NULL AND content != ''"
        )
    logger.info(f"Document chunks to embed: {total}")

    if total == 0:
        return

    processed = 0
    while True:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, content FROM document_chunks "
                "WHERE embedding IS NULL AND content IS NOT NULL AND content != '' "
                "ORDER BY id LIMIT $1",
                batch_size,
            )

        if not rows:
            break

        texts = [row["content"] for row in rows]
        ids = [row["id"] for row in rows]

        start = time.time()
        embeddings = embed_batch(model, texts)
        elapsed = time.time() - start

        async with pool.acquire() as conn:
            for chunk_id, emb in zip(ids, embeddings):
                await conn.execute(
                    "UPDATE document_chunks SET embedding = $1 WHERE id = $2",
                    str(emb), chunk_id,
                )

        processed += len(rows)
        logger.info(f"Document chunks: {processed}/{total} ({elapsed:.1f}s for {len(rows)} chunks)")


async def backfill_research_chunks(pool: asyncpg.Pool, model, batch_size: int):
    """Chunk and embed completed research sessions that have no chunks yet."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT rs.id, rs.workspace_id, rs.content
               FROM research_sessions rs
               WHERE rs.status = 'completed'
                 AND rs.content IS NOT NULL AND rs.content != ''
                 AND NOT EXISTS (SELECT 1 FROM research_chunks rc WHERE rc.research_id = rs.id)
               ORDER BY rs.id"""
        )
    logger.info(f"Research sessions to chunk+embed: {len(rows)}")

    for i, row in enumerate(rows):
        research_id = row["id"]
        workspace_id = row["workspace_id"]
        content = row["content"]

        # Split into chunks using simple paragraph splitting
        import re
        paragraphs = re.split(r'\n\s*\n', content)
        chunks = []
        current_chunk = []
        current_len = 0
        chunk_idx = 0
        char_pos = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if current_len + len(para) > 1500 and current_chunk:
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append({
                    "chunk_index": chunk_idx,
                    "char_start": char_pos - current_len,
                    "char_end": char_pos,
                    "content": chunk_text,
                })
                chunk_idx += 1
                current_chunk = []
                current_len = 0
            current_chunk.append(para)
            current_len += len(para) + 2
            char_pos += len(para) + 2

        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append({
                "chunk_index": chunk_idx,
                "char_start": char_pos - current_len,
                "char_end": char_pos,
                "content": chunk_text,
            })

        if not chunks:
            continue

        texts = [c["content"] for c in chunks]
        start = time.time()
        embeddings = embed_batch(model, texts)
        elapsed = time.time() - start

        async with pool.acquire() as conn:
            for chunk, emb in zip(chunks, embeddings):
                await conn.execute(
                    """INSERT INTO research_chunks (workspace_id, research_id, chunk_index, char_start, char_end, content, search_vector, embedding)
                       VALUES ($1, $2, $3, $4, $5, $6, to_tsvector('english', $6), $7)
                       ON CONFLICT (research_id, chunk_index) DO UPDATE
                       SET content = EXCLUDED.content, search_vector = EXCLUDED.search_vector, embedding = EXCLUDED.embedding""",
                    workspace_id, research_id, chunk["chunk_index"],
                    chunk["char_start"], chunk["char_end"], chunk["content"],
                    str(emb),
                )

        logger.info(f"Research {i+1}/{len(rows)}: session {research_id} → {len(chunks)} chunks ({elapsed:.1f}s)")


async def main():
    parser = argparse.ArgumentParser(description="Backfill embeddings for existing chunks")
    parser.add_argument("--batch-size", type=int, default=50, help="Chunks per batch")
    parser.add_argument("--meetings-only", action="store_true")
    parser.add_argument("--documents-only", action="store_true")
    parser.add_argument("--research-only", action="store_true")
    args = parser.parse_args()

    model = load_model()
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=3)

    try:
        if args.research_only:
            await backfill_research_chunks(pool, model, args.batch_size)
        else:
            if not args.documents_only:
                await backfill_meeting_chunks(pool, model, args.batch_size)
            if not args.meetings_only:
                await backfill_document_chunks(pool, model, args.batch_size)
            await backfill_research_chunks(pool, model, args.batch_size)
    finally:
        await pool.close()

    logger.info("Backfill complete")


if __name__ == "__main__":
    asyncio.run(main())
