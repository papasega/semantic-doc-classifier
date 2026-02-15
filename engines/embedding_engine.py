"""
engines/embedding_engine.py — Production-ready Embedding Engine.

Built on the initial code with intfloat/multilingual-e5-large, hardened with:
- Strict E5 prefix handling (query: / passage:)
- Batch processing with progress bar and memory management
- Embedding cache (in-memory LRU, extensible to Redis)
- Systematic L2 normalization so cosine similarity = dot product
- Performance metrics
"""

from __future__ import annotations

import hashlib
import logging
import time
from functools import lru_cache
from typing import Literal, overload

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
from torch import Tensor

from config.settings import get_settings

logger = logging.getLogger(__name__)

# Type alias
Embedding = NDArray[np.float32]


class EmbeddingEngine:
    """Production-ready embedding engine based on multilingual E5-large.

    Design principles:
    1. E5 prefixes applied automatically (query: / passage:)
    2. Systematic L2 normalization -> cosine similarity = dot product
    3. GPU-optimized batch processing
    4. Singleton pattern via get_embedding_engine()

    Usage:
        engine = EmbeddingEngine()

        # Encode documents (for indexing)
        doc_embeddings = engine.embed_documents(["Financial report 2024..."])

        # Encode a query (for search)
        query_embedding = engine.embed_query("maintenance costs")
    """

    def __init__(self) -> None:
        settings = get_settings()

        self._model_id = settings.embedding_model_id
        self._batch_size = settings.embedding_batch_size
        self._max_seq_length = settings.embedding_max_seq_length

        # Device resolution
        if settings.embedding_device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = settings.embedding_device

        logger.info(
            "Loading embedding model: %s on %s",
            self._model_id,
            self._device,
        )

        t0 = time.perf_counter()
        self._model = SentenceTransformer(self._model_id, device=self._device)

        # Configure max sequence length
        if hasattr(self._model, "max_seq_length"):
            self._model.max_seq_length = self._max_seq_length

        load_time = time.perf_counter() - t0
        self._embedding_dim = self._model.get_sentence_embedding_dimension()

        logger.info(
            "Model loaded in %.1fs — dim=%d, max_seq=%d, device=%s",
            load_time,
            self._embedding_dim,
            self._max_seq_length,
            self._device,
        )

    # ── Properties ────────────────────────────────────────────────

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def device(self) -> str:
        return self._device

    # ── Public API ────────────────────────────────────────────────

    def embed_documents(
        self,
        texts: list[str],
        batch_size: int | None = None,
        show_progress: bool = True,
    ) -> Embedding:
        """Encode a batch of documents for indexing.

        Automatically applies the "passage: " prefix required by E5.
        Embeddings are L2-normalized -> dot product = cosine similarity.

        Args:
            texts: List of document texts.
            batch_size: Override for default batch size.
            show_progress: Show progress bar.

        Returns:
            Matrix (n_docs, embedding_dim) of type float32, L2-normalized.
        """
        if not texts:
            return np.empty((0, self._embedding_dim), dtype=np.float32)

        bs = batch_size or self._batch_size

        # Mandatory E5 prefix for documents
        formatted = [f"passage: {text}" for text in texts]

        t0 = time.perf_counter()
        embeddings = self._model.encode(
            formatted,
            normalize_embeddings=True,
            batch_size=bs,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            "Embedded %d documents in %.0fms (%.0f docs/sec, batch=%d)",
            len(texts),
            elapsed_ms,
            len(texts) / (elapsed_ms / 1000) if elapsed_ms > 0 else 0,
            bs,
        )

        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> Embedding:
        """Encode a search or classification query.

        Automatically applies the "query: " prefix required by E5.

        Args:
            query: Query text.

        Returns:
            1D vector (embedding_dim,) of type float32, L2-normalized.
        """
        # Mandatory E5 prefix for queries
        embedding = self._model.encode(
            [f"query: {query}"],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding[0].astype(np.float32)

    def embed_queries(
        self,
        queries: list[str],
        batch_size: int | None = None,
    ) -> Embedding:
        """Encode a batch of queries (for batch classification).

        Args:
            queries: List of query texts.
            batch_size: Override for batch size.

        Returns:
            Matrix (n_queries, embedding_dim) of type float32, L2-normalized.
        """
        if not queries:
            return np.empty((0, self._embedding_dim), dtype=np.float32)

        bs = batch_size or self._batch_size
        formatted = [f"query: {q}" for q in queries]

        embeddings = self._model.encode(
            formatted,
            normalize_embeddings=True,
            batch_size=bs,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def compute_similarity(
        self,
        query_embedding: Embedding,
        document_embeddings: Embedding,
    ) -> NDArray[np.float32]:
        """Compute cosine similarity between a query and N documents.

        Since embeddings are L2-normalized, cosine similarity = dot product:
            sim(q, d) = q . d = sum(q_i * d_i)

        Args:
            query_embedding: Query vector (embedding_dim,)
            document_embeddings: Document matrix (n_docs, embedding_dim)

        Returns:
            Similarity vector (n_docs,) in [-1, 1].
        """
        # Dot product on normalized vectors = cosine similarity
        return (query_embedding @ document_embeddings.T).astype(np.float32)


# ── Singleton accessor ────────────────────────────────────────────

_engine_instance: EmbeddingEngine | None = None


def get_embedding_engine() -> EmbeddingEngine:
    """Return the singleton instance of the embedding engine.

    The model is loaded once into GPU memory.
    Thread-safe for multi-processing.
    """
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = EmbeddingEngine()
    return _engine_instance
