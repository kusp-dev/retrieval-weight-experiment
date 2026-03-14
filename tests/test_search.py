"""Tests for hybrid search (BM25 + dense + RRF).

Uses synthetic embeddings (random vectors) to test search mechanics
without loading the actual model.
"""

import numpy as np
import pytest

from src.db.schema import init_db
from src.embeddings.embedder import vector_to_blob
from src.search.hybrid import HybridSearchEngine, SearchResult


@pytest.fixture
def db_with_skills():
    """Database with 5 test skills and synthetic embeddings."""
    conn = init_db(":memory:")
    rng = np.random.default_rng(42)

    skills = [
        ("s1", "research", "Neural Retrieval", "Dense retrieval using transformer neural networks"),
        ("s2", "research", "BM25 Basics", "BM25 term frequency inverse document frequency scoring"),
        (
            "s3",
            "synthesis",
            "RAG Pipeline",
            "Retrieval augmented generation combining search and LLM",
        ),
        (
            "s4",
            "evaluation",
            "MTEB Benchmark",
            "Massive text embedding benchmark evaluation metrics",
        ),
        (
            "s5",
            "operational",
            "Vector Databases",
            "Vector database indexing with HNSW approximate search",
        ),
    ]

    for sid, domain, title, content in skills:
        conn.execute(
            "INSERT INTO skills (skill_id, domain, title, content) VALUES (?, ?, ?, ?)",
            (sid, domain, title, content),
        )
        # Random 1024-dim embedding (normalized)
        vec = rng.standard_normal(1024).astype(np.float32)
        vec /= np.linalg.norm(vec)
        conn.execute(
            "INSERT INTO skill_embeddings (skill_id, embedding) VALUES (?, ?)",
            (sid, vector_to_blob(vec)),
        )

    conn.commit()
    yield conn
    conn.close()


class TestBM25Search:
    def test_finds_matching_skills(self, db_with_skills):
        engine = HybridSearchEngine(conn=db_with_skills)
        results = engine.search_bm25("neural retrieval transformer", top_k=5)
        skill_ids = [r[0] for r in results]
        assert "s1" in skill_ids  # "Neural Retrieval" matches best

    def test_no_results_for_unrelated_query(self, db_with_skills):
        engine = HybridSearchEngine(conn=db_with_skills)
        results = engine.search_bm25("quantum_physics_unrelated_xyz", top_k=5)
        assert len(results) == 0

    def test_respects_top_k(self, db_with_skills):
        engine = HybridSearchEngine(conn=db_with_skills)
        results = engine.search_bm25("search retrieval", top_k=2)
        assert len(results) <= 2


class TestDenseSearch:
    def test_returns_results(self, db_with_skills):
        engine = HybridSearchEngine(conn=db_with_skills)
        query_vec = np.random.default_rng(99).standard_normal(1024).astype(np.float32)
        query_vec /= np.linalg.norm(query_vec)

        results = engine.search_dense(query_vec, top_k=3)
        assert len(results) == 3
        assert all(isinstance(r[1], float) for r in results)

    def test_results_sorted_by_similarity(self, db_with_skills):
        engine = HybridSearchEngine(conn=db_with_skills)
        query_vec = np.random.default_rng(99).standard_normal(1024).astype(np.float32)
        query_vec /= np.linalg.norm(query_vec)

        results = engine.search_dense(query_vec, top_k=5)
        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)


class TestRRFFusion:
    def test_fuses_overlapping_results(self, db_with_skills):
        engine = HybridSearchEngine(conn=db_with_skills, rrf_k=60)

        bm25 = [("s1", 10.0), ("s2", 8.0), ("s3", 5.0)]
        dense = [("s1", 0.9), ("s3", 0.85), ("s4", 0.7)]

        scores = engine.fuse_rrf(bm25, dense)

        # s1 appears in both lists → highest score
        assert scores["s1"] > scores["s2"]
        assert scores["s1"] > scores["s4"]
        # s3 also in both
        assert scores["s3"] > scores["s4"]

    def test_handles_disjoint_results(self, db_with_skills):
        engine = HybridSearchEngine(conn=db_with_skills)

        bm25 = [("s1", 10.0)]
        dense = [("s2", 0.9)]

        scores = engine.fuse_rrf(bm25, dense)
        assert "s1" in scores
        assert "s2" in scores


class TestHybridSearch:
    def test_full_pipeline(self, db_with_skills):
        engine = HybridSearchEngine(conn=db_with_skills)

        query_vec = np.random.default_rng(99).standard_normal(1024).astype(np.float32)
        query_vec /= np.linalg.norm(query_vec)

        results = engine.search("neural retrieval", query_vec, top_k=3)
        assert len(results) <= 3
        assert all(isinstance(r, SearchResult) for r in results)

    def test_results_have_scores(self, db_with_skills):
        engine = HybridSearchEngine(conn=db_with_skills)

        query_vec = np.random.default_rng(99).standard_normal(1024).astype(np.float32)
        query_vec /= np.linalg.norm(query_vec)

        results = engine.search("retrieval search", query_vec, top_k=5)
        for r in results:
            assert r.rrf_score >= 0
            assert r.final_score >= 0

    def test_with_retrieval_weights(self, db_with_skills):
        engine = HybridSearchEngine(conn=db_with_skills)

        query_vec = np.random.default_rng(99).standard_normal(1024).astype(np.float32)
        query_vec /= np.linalg.norm(query_vec)

        results = engine.search(
            "retrieval",
            query_vec,
            top_k=3,
            retrieval_weights=(0.1, 0.2, 0.7),
        )
        # With all dimension scores at 0 (not set), final_score = 0
        # This tests the pipeline doesn't crash with weights
        assert len(results) <= 3

    def test_cache_invalidation(self, db_with_skills):
        engine = HybridSearchEngine(conn=db_with_skills)

        query_vec = np.random.default_rng(99).standard_normal(1024).astype(np.float32)
        query_vec /= np.linalg.norm(query_vec)

        # First search loads cache
        engine.search("test", query_vec, top_k=3)
        assert engine._cache_loaded

        # Invalidate
        engine.invalidate_cache()
        assert not engine._cache_loaded
