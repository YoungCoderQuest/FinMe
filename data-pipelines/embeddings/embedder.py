"""
Embedder — FinMe

Fixes over original:
- Lazy loading: model loads on first use, not at import time
- Offline mode: uses local cache, never hits HuggingFace after first download
- Single instance: model is cached after first load (no repeated loading)
"""

import os
from sentence_transformers import SentenceTransformer

# Force offline mode — use local cache only after first download
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model = None  # lazy singleton


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def generate_embeddings(chunks: list[str]):
    return _get_model().encode(
        chunks,
        convert_to_numpy=True
    )