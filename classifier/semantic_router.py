"""
classifier/semantic_router.py — Zero-Fragility Semantic Router.

══════════════════════════════════════════════════════════════════
 CORE PRINCIPLE
══════════════════════════════════════════════════════════════════

 NO if/else logic on document content.
 NO keyword matching, regex, or deterministic heuristics.

 Classification relies SOLELY on vector geometry:

     1. Each class = centroid in embedding space
     2. Document -> embedding via E5
     3. Classification = argmax(cosine_similarity(doc, centroids))

 Mathematically, with L2-normalized vectors:

     sim(q, c_i) = q . c_i = sum(q_j * c_ij)    for all i in {1,...,K}
     class = argmax_i { sim(q, c_i) }

══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
import time
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from config.routes import DocumentType, RoutePrototype, get_routes
from config.settings import get_settings
from engines.embedding_engine import EmbeddingEngine, get_embedding_engine
from classifier.models import ClassificationCandidate, ClassificationResult

logger = logging.getLogger(__name__)

Embedding = NDArray[np.float32]


class SemanticRouter:
    """Semantic router for document classification.

    Internal architecture:
    +─────────────────────────────────────────────────────────+
    | Init:                                                   |
    |   Routes (NL descriptions) -> embed_documents()         |
    |     -> Embedding matrix -> mean() -> Centroids (K, d)   |
    |                                                         |
    | Classify:                                               |
    |   Document text -> embed_query()                        |
    |     -> Query vector (d,) @ Centroids.T -> scores (K,)   |
    |     -> argsort -> top-k -> threshold -> Result          |
    +─────────────────────────────────────────────────────────+

    The router is stateless after init: thread-safe for serving.
    """

    def __init__(
        self,
        embedding_engine: EmbeddingEngine | None = None,
        routes: list[RoutePrototype] | None = None,
        confidence_threshold: float | None = None,
    ) -> None:
        """Initialize the router by pre-computing centroids.

        Args:
            embedding_engine: Embedding engine instance.
                If None, uses the global singleton.
            routes: Semantic route definitions.
                If None, loads default routes from config/routes.py.
            confidence_threshold: Confidence threshold.
                If None, uses the value from settings.
        """
        settings = get_settings()

        self._engine = embedding_engine or get_embedding_engine()
        self._routes = routes or get_routes()
        self._threshold = confidence_threshold or settings.confidence_threshold
        self._top_k_default = settings.classification_top_k
        self._text_truncation = settings.text_truncation_chars

        # Pre-compute centroids
        self._route_types: list[DocumentType] = []
        self._centroids: Embedding | None = None
        self._build_centroids()

    # ── Public API ────────────────────────────────────────────────

    def classify(self, text: str, top_k: int | None = None) -> ClassificationResult:
        """Classify a document via semantic routing.

        Args:
            text: Extracted document text.
            top_k: Number of candidate classes (default: from settings).

        Returns:
            ClassificationResult with type, confidence, top-k and metrics.
        """
        k = top_k or self._top_k_default
        t0 = time.perf_counter()

        # Text truncation (E5 has a limited context window)
        truncated = text[: self._text_truncation]

        # Query embedding
        query_vec = self._engine.embed_query(truncated)

        # Cosine similarity via dot product (normalized vectors)
        # query_vec: (d,) @ centroids.T: (d, K) -> (K,)
        similarities = self._engine.compute_similarity(query_vec, self._centroids)

        # Results
        result = self._rank_results(similarities, k, t0)

        logger.debug(
            "Classification: %s (conf=%.3f, margin=%.3f, latency=%.1fms)",
            result.document_type.value,
            result.confidence,
            result.margin,
            result.latency_ms,
        )

        return result

    def classify_batch(
        self,
        texts: list[str],
        top_k: int | None = None,
        batch_size: int | None = None,
    ) -> list[ClassificationResult]:
        """GPU-optimized batch classification.

        Encodes all texts in a single GPU call, then computes
        the similarity matrix via matrix multiplication.

        On an A100 with E5-large and batch_size=32:
            ~400-600 documents/second

        Args:
            texts: List of texts to classify.
            top_k: Number of candidate classes per document.
            batch_size: GPU batch size.

        Returns:
            List of ClassificationResult (same order as texts).
        """
        if not texts:
            return []

        k = top_k or self._top_k_default
        t0 = time.perf_counter()

        # Truncation
        truncated = [t[: self._text_truncation] for t in texts]

        # Batch embedding: all queries in a single GPU call
        query_matrix = self._engine.embed_queries(truncated, batch_size=batch_size)

        # Similarity matrix: (n_docs, K)
        # query_matrix: (n_docs, d) @ centroids.T: (d, K) -> (n_docs, K)
        sim_matrix = query_matrix @ self._centroids.T

        # Per-document results
        total_elapsed = time.perf_counter() - t0
        per_doc_ms = (total_elapsed * 1000) / len(texts) if texts else 0

        results = []
        for i in range(len(texts)):
            result = self._rank_results(sim_matrix[i], k, t0, override_latency_ms=per_doc_ms)
            results.append(result)

        throughput = len(texts) / total_elapsed if total_elapsed > 0 else 0
        logger.info(
            "Batch classified: %d docs in %.0fms (%.0f docs/sec)",
            len(texts),
            total_elapsed * 1000,
            throughput,
        )

        return results

    # ── Introspection ─────────────────────────────────────────────

    @property
    def route_types(self) -> list[DocumentType]:
        """Configured document types (excluding INCONNU)."""
        return list(self._route_types)

    @property
    def n_routes(self) -> int:
        return len(self._route_types)

    @property
    def embedding_dim(self) -> int:
        return self._centroids.shape[1] if self._centroids is not None else 0

    @property
    def confidence_threshold(self) -> float:
        return self._threshold

    def get_centroid(self, doc_type: DocumentType) -> Embedding | None:
        """Return the centroid for a type (for debugging/visualization)."""
        if doc_type in self._route_types:
            idx = self._route_types.index(doc_type)
            return self._centroids[idx].copy()
        return None

    def explain(self, text: str) -> dict:
        """Return a detailed explanation of the classification.

        Useful for debugging and human validation.
        """
        result = self.classify(text, top_k=len(self._route_types))
        return {
            "predicted": result.document_type.value,
            "confidence": result.confidence,
            "threshold": self._threshold,
            "is_confident": result.is_confident,
            "margin": result.margin,
            "all_scores": {
                c.document_type.value: round(c.confidence, 4)
                for c in result.top_k
            },
            "model": result.embedding_model,
            "text_preview": text[:200] + "..." if len(text) > 200 else text,
        }

    # ── Internal ──────────────────────────────────────────────────

    def _build_centroids(self) -> None:
        """Pre-compute centroids for semantic routes.

        For each route:
        1. Encode all descriptions -> matrix (n_desc, d)
        2. Mean -> vector (d,)
        3. L2 normalization -> unit centroid

        Result: matrix (K, d) where K = number of types.
        """
        t0 = time.perf_counter()
        centroids = []

        for route in self._routes:
            # Encode descriptions as documents (passage:)
            desc_embeddings = self._engine.embed_documents(
                list(route.descriptions),
                batch_size=len(route.descriptions),
                show_progress=False,
            )

            # Centroid = L2-normalized mean
            centroid = np.mean(desc_embeddings, axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 1e-8:
                centroid = centroid / norm

            centroids.append(centroid)
            self._route_types.append(route.document_type)

        self._centroids = np.stack(centroids).astype(np.float32)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "Centroids computed: %d routes, dim=%d, in %.0fms",
            len(self._route_types),
            self._centroids.shape[1],
            elapsed_ms,
        )

    def _rank_results(
        self,
        similarities: NDArray[np.float32],
        top_k: int,
        t0: float,
        override_latency_ms: float | None = None,
    ) -> ClassificationResult:
        """Build the ClassificationResult from similarity scores.

        Args:
            similarities: Vector (K,) of cosine similarities.
            top_k: Number of candidates to return.
            t0: Start timestamp for latency computation.
            override_latency_ms: Latency override (for batch mode).
        """
        ranked_indices = np.argsort(similarities)[::-1][:top_k]

        candidates = [
            ClassificationCandidate(
                document_type=self._route_types[idx],
                confidence=round(float(similarities[idx]), 4),
            )
            for idx in ranked_indices
        ]

        best = candidates[0]

        # Confidence threshold: if best score is too low -> INCONNU
        predicted_type = best.document_type
        if best.confidence < self._threshold:
            predicted_type = DocumentType.INCONNU

        latency = override_latency_ms or ((time.perf_counter() - t0) * 1000)

        return ClassificationResult(
            document_type=predicted_type,
            confidence=best.confidence,
            top_k=candidates,
            embedding_model=self._engine.model_id,
            confidence_threshold=self._threshold,
            latency_ms=round(latency, 2),
        )
