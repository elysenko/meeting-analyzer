"""
Embedding model singleton and batch embed helper.

Extracted from main_live.py top-level globals.
No app or DB dependencies — pure compute.
"""

import logging
import os

from config import EMBEDDING_MODEL_NAME

logger = logging.getLogger("meeting-analyzer")

_embedding_model = None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from fastembed import TextEmbedding
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        cache_dir = os.getenv("FASTEMBED_CACHE_PATH", "/models/fastembed")
        _embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME, cache_dir=cache_dir)
        logger.info("Embedding model loaded")
    return _embedding_model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a batch of texts using fastembed."""
    model = _get_embedding_model()
    embeddings = list(model.embed(texts))
    return [e.tolist() for e in embeddings]
