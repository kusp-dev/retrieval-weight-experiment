"""
Hybrid search: BM25 (SQLite FTS5) + dense (Qwen3-Embedding) + RRF fusion.

Two-stage pipeline:
  1. Search: BM25 + dense → RRF fusion → candidate set
  2. Score: Apply retrieval weights (recency, importance, relevance) → final ranking

Stage 1 is the retrieval mechanism. Stage 2 is what Thompson Sampling optimizes.
"""

import re
import sqlite3
from dataclasses import dataclass, field

import numpy as np

from src.embeddings.embedder import (
    cosine_similarity_batch,
    load_all_skill_embeddings,
)


@dataclass
class SearchResult:
    """A single search result with scoring breakdown."""

    skill_id: str
    title: str = ""
    bm25_score: float = 0.0
    dense_score: float = 0.0
    rrf_score: float = 0.0
    final_score: float = 0.0
    # Dimension scores (set externally based on task context)
    recency_score: float = 0.0
    importance_score: float = 0.0
    relevance_score: float = 0.0


@dataclass
class HybridSearchEngine:
    """BM25 + dense vector search with Reciprocal Rank Fusion.

    Caches skill embeddings in memory on first search.
    """

    conn: sqlite3.Connection
    rrf_k: int = 60
    bm25_weight: float = 0.4
    dense_weight: float = 0.6
    dimensions: int = 1024

    # Cached state
    _skill_ids: list[str] = field(default_factory=list, init=False, repr=False)
    _embedding_matrix: np.ndarray = field(
        default_factory=lambda: np.empty(0), init=False, repr=False
    )
    _cache_loaded: bool = field(default=False, init=False, repr=False)

    def _ensure_cache(self) -> None:
        """Load skill embeddings into memory if not already cached."""
        if self._cache_loaded:
            return
        self._skill_ids, self._embedding_matrix = load_all_skill_embeddings(
            self.conn, self.dimensions
        )
        self._cache_loaded = True

    def invalidate_cache(self) -> None:
        """Force reload of skill embeddings on next search."""
        self._cache_loaded = False

    @staticmethod
    def _sanitize_fts5_query(query: str) -> str | None:
        """Sanitize a query string for FTS5 MATCH.

        Raw text can contain FTS5 operators (colons, hyphens, parentheses)
        that cause syntax errors. This strips non-word characters and quotes
        each term individually, joined with OR for broad matching.
        """
        cleaned = re.sub(r"[^\w\s]", " ", query)
        words = [f'"{w}"' for w in cleaned.split() if len(w) > 1]
        return " OR ".join(words) if words else None

    def search_bm25(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """BM25 keyword search via FTS5.

        Returns list of (skill_id, bm25_rank_score) ordered by relevance.
        """
        sanitized = self._sanitize_fts5_query(query)
        if sanitized is None:
            return []

        rows = self.conn.execute(
            """SELECT skill_id, rank
               FROM skills_fts
               WHERE skills_fts MATCH ?
               ORDER BY rank
               LIMIT ?""",
            (sanitized, top_k),
        ).fetchall()

        # FTS5 rank is negative (lower = better). Convert to positive score.
        return [(row["skill_id"], -row["rank"]) for row in rows]

    def search_dense(self, query_embedding: np.ndarray, top_k: int = 20) -> list[tuple[str, float]]:
        """Dense vector search using cosine similarity.

        Returns list of (skill_id, cosine_similarity) ordered by similarity.
        """
        self._ensure_cache()

        if len(self._skill_ids) == 0:
            return []

        similarities = cosine_similarity_batch(query_embedding, self._embedding_matrix)

        # Get top-k indices
        if len(similarities) <= top_k:
            top_indices = np.argsort(similarities)[::-1]
        else:
            top_indices = np.argpartition(similarities, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        return [(self._skill_ids[i], float(similarities[i])) for i in top_indices]

    def fuse_rrf(
        self,
        bm25_results: list[tuple[str, float]],
        dense_results: list[tuple[str, float]],
    ) -> dict[str, float]:
        """Reciprocal Rank Fusion of BM25 and dense results.

        RRF(d) = Σ (weight / (k + rank(d)))

        Returns dict of skill_id → RRF score.
        """
        scores: dict[str, float] = {}

        for rank, (skill_id, _) in enumerate(bm25_results):
            rrf = self.bm25_weight / (self.rrf_k + rank + 1)
            scores[skill_id] = scores.get(skill_id, 0.0) + rrf

        for rank, (skill_id, _) in enumerate(dense_results):
            rrf = self.dense_weight / (self.rrf_k + rank + 1)
            scores[skill_id] = scores.get(skill_id, 0.0) + rrf

        return scores

    def search(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 10,
        retrieval_weights: tuple[float, float, float] | None = None,
    ) -> list[SearchResult]:
        """Full hybrid search pipeline.

        Args:
            query: text query for BM25
            query_embedding: dense vector for semantic search
            top_k: number of results to return
            retrieval_weights: (w_recency, w_importance, w_relevance) for stage 2 scoring.
                              If None, returns results ranked by RRF score only (stage 1).

        Returns:
            List of SearchResult ordered by final_score (descending).
        """
        # Stage 1: Search + RRF fusion
        fetch_k = max(top_k * 3, 20)  # over-fetch for fusion
        bm25_results = self.search_bm25(query, top_k=fetch_k)
        dense_results = self.search_dense(query_embedding, top_k=fetch_k)
        rrf_scores = self.fuse_rrf(bm25_results, dense_results)

        # Build lookup dicts for individual scores
        bm25_lookup = dict(bm25_results)
        dense_lookup = dict(dense_results)

        # Build result objects
        results = []
        for skill_id, rrf_score in rrf_scores.items():
            result = SearchResult(
                skill_id=skill_id,
                bm25_score=bm25_lookup.get(skill_id, 0.0),
                dense_score=dense_lookup.get(skill_id, 0.0),
                rrf_score=rrf_score,
            )

            # Look up title
            row = self.conn.execute(
                "SELECT title FROM skills WHERE skill_id = ?", (skill_id,)
            ).fetchone()
            if row:
                result.title = row["title"]

            results.append(result)

        # Stage 2: Apply retrieval weights if provided
        if retrieval_weights is not None:
            w_rec, w_imp, w_rel = retrieval_weights
            for r in results:
                # relevance_score is the RRF-fused retrieval score (how well it matched)
                # recency_score and importance_score are set externally per skill
                r.final_score = (
                    w_rec * r.recency_score + w_imp * r.importance_score + w_rel * r.relevance_score
                )
        else:
            for r in results:
                r.final_score = r.rrf_score

        # Sort by final score descending, take top_k
        results.sort(key=lambda x: x.final_score, reverse=True)
        return results[:top_k]
