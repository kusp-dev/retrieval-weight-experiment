"""
End-to-end integration test: DB → embeddings → search → bandit → feedback loop.

Uses synthetic embeddings to avoid model download. Tests the full pipeline
that will run during the actual experiment.
"""

import json

import numpy as np
import pytest

from src.db.schema import (
    init_bandit_state,
    init_db,
    log_event,
    seed_weight_presets,
)
from src.embeddings.embedder import (
    blob_to_vector,
    cosine_similarity,
    store_skill_embedding,
    vector_to_blob,
)
from src.experiment.bandit import ThompsonSamplingBandit, normalize_likert_to_reward
from src.search.hybrid import HybridSearchEngine

PRESETS = {
    "balanced": [0.330, 0.330, 0.340],
    "recency_heavy": [0.600, 0.200, 0.200],
    "importance_heavy": [0.150, 0.700, 0.150],
    "relevance_heavy": [0.200, 0.150, 0.650],
    "relevance_importance": [0.150, 0.350, 0.500],
}


@pytest.fixture
def experiment_db():
    """Fully initialized experiment database with skills and presets."""
    conn = init_db(":memory:")
    rng = np.random.default_rng(42)

    # Seed weight presets
    seed_weight_presets(conn, PRESETS)

    # Initialize bandit for conditions 2, 3, 4 (not control)
    init_bandit_state(conn, [2, 3, 4], list(PRESETS.keys()))

    # Insert test skills with synthetic embeddings
    skills = [
        (
            "ml-001",
            "research",
            "Gradient Descent",
            "Optimization via gradient descent backpropagation learning rate",
        ),
        (
            "ml-002",
            "research",
            "Attention Mechanism",
            "Self-attention transformer multi-head attention mechanism",
        ),
        ("ag-001", "synthesis", "ReAct Pattern", "Reasoning and acting agent loop with tool use"),
        (
            "ag-002",
            "synthesis",
            "Chain of Thought",
            "Step by step reasoning chain of thought prompting",
        ),
        (
            "rt-001",
            "evaluation",
            "BM25 Algorithm",
            "Term frequency inverse document frequency BM25 ranking",
        ),
    ]

    for sid, domain, title, content in skills:
        conn.execute(
            "INSERT INTO skills (skill_id, domain, title, content) VALUES (?, ?, ?, ?)",
            (sid, domain, title, content),
        )
        vec = rng.standard_normal(1024).astype(np.float32)
        vec /= np.linalg.norm(vec)
        store_skill_embedding(conn, sid, vec)

    # Insert a test task
    conn.execute(
        "INSERT INTO tasks (task_id, theme, title, description) VALUES (?, ?, ?, ?)",
        (
            "t-001",
            "ML Fundamentals",
            "Explain SGD",
            "Explain stochastic gradient descent and its variants",
        ),
    )
    conn.commit()

    yield conn
    conn.close()


class TestFullPipeline:
    """Simulates one complete episode: task → search → execute → feedback → bandit update."""

    def test_single_episode_condition_2(self, experiment_db):
        """Condition 2: TS + Likert feedback, explanations NOT embedded."""
        conn = experiment_db
        rng = np.random.default_rng(42)
        condition_id = 2

        # 1. Bandit selects a weight preset
        bandit = ThompsonSamplingBandit(conn, condition_id, rng=rng)
        preset_id = bandit.select_arm()
        assert preset_id in PRESETS

        # 2. Get the weights for the selected preset
        row = conn.execute(
            "SELECT w_recency, w_importance, w_relevance FROM weight_presets WHERE preset_id = ?",
            (preset_id,),
        ).fetchone()
        weights = (row["w_recency"], row["w_importance"], row["w_relevance"])

        # 3. Search for relevant skills
        query = "stochastic gradient descent optimization"
        query_vec = rng.standard_normal(1024).astype(np.float32)
        query_vec /= np.linalg.norm(query_vec)

        engine = HybridSearchEngine(conn=conn)
        results = engine.search(query, query_vec, top_k=3, retrieval_weights=weights)

        # 4. Record the episode
        conn.execute(
            """INSERT INTO episodes (condition_id, task_id, preset_id, task_order)
               VALUES (?, ?, ?, ?)""",
            (condition_id, "t-001", preset_id, 1),
        )
        conn.commit()
        episode_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        # 5. Record retrieval results
        for rank, r in enumerate(results):
            conn.execute(
                """INSERT INTO retrieval_results
                   (episode_id, skill_id, rank, bm25_score, dense_score, rrf_score, final_score)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    episode_id,
                    r.skill_id,
                    rank,
                    r.bm25_score,
                    r.dense_score,
                    r.rrf_score,
                    r.final_score,
                ),
            )

        # 6. Simulate LLM execution (skip actual LLM call)
        conn.execute(
            "UPDATE episodes SET llm_response = ?,"
            " success = 1,"
            " completed_at = datetime('now')"
            " WHERE episode_id = ?",
            ("SGD is an optimization algorithm...", episode_id),
        )

        # 7. Record dimension feedback (Likert 1-5)
        conn.execute(
            """INSERT INTO feedback
               (episode_id, rating_recency, rating_importance,
                rating_relevance, explanation)
               VALUES (?, ?, ?, ?, ?)""",
            (
                episode_id,
                3,
                4,
                5,
                "Relevance was excellent — the gradient descent skill was highly applicable.",
            ),
        )
        conn.commit()

        # 8. Update bandit
        reward = normalize_likert_to_reward(3, 4, 5)
        bandit.update(preset_id, reward)

        # 9. Verify state
        summary = bandit.get_summary()
        assert summary["total_pulls"] == 1

        # Verify episode was recorded
        ep = conn.execute("SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)).fetchone()
        assert ep["success"] == 1
        assert ep["condition_id"] == condition_id

        # Verify feedback
        fb = conn.execute("SELECT * FROM feedback WHERE episode_id = ?", (episode_id,)).fetchone()
        assert fb["rating_relevance"] == 5

        # Verify retrieval results
        rr = conn.execute(
            "SELECT COUNT(*) as n FROM retrieval_results WHERE episode_id = ?",
            (episode_id,),
        ).fetchone()
        assert rr["n"] == len(results)

    def test_control_condition(self, experiment_db):
        """Condition 1: Fixed equal weights, no feedback, no bandit."""
        conn = experiment_db

        # Control always uses 'equal' preset
        preset_id = "balanced"
        row = conn.execute(
            "SELECT w_recency, w_importance, w_relevance FROM weight_presets WHERE preset_id = ?",
            (preset_id,),
        ).fetchone()
        weights = (row["w_recency"], row["w_importance"], row["w_relevance"])

        # Verify weights are approximately equal (balanced preset)
        assert abs(weights[0] - weights[1]) < 0.02
        assert abs(weights[1] - weights[2]) < 0.02

        # Record episode without feedback
        conn.execute(
            "INSERT INTO episodes"
            " (condition_id, task_id, preset_id, task_order)"
            " VALUES (1, 't-001', 'balanced', 1)"
        )
        conn.commit()

    def test_multiple_episodes_bandit_learns(self, experiment_db):
        """Verify bandit posteriors diverge after multiple episodes."""
        conn = experiment_db
        rng = np.random.default_rng(42)
        bandit = ThompsonSamplingBandit(conn, condition_id=2, rng=rng)

        # Simulate 50 episodes where relevance_heavy consistently gets better feedback
        for i in range(50):
            selected = bandit.select_arm()

            # Simulate: relevance_heavy tasks get great feedback
            if selected == "relevance_heavy":
                reward = normalize_likert_to_reward(3, 4, 5)  # 0.75
            elif selected == "relevance_importance":
                reward = normalize_likert_to_reward(3, 3, 4)  # 0.583
            else:
                reward = normalize_likert_to_reward(2, 2, 3)  # 0.333

            bandit.update(selected, reward)

        summary = bandit.get_summary()
        best = summary["best_arm"]

        # After 50 episodes with this reward structure, relevance_heavy should lead
        assert best == "relevance_heavy"

        # Log it
        log_event(conn, "test", f"Bandit converged to {best}", metadata=json.dumps(summary))


class TestEmbeddingRoundTrip:
    def test_vector_serialization(self):
        """Vectors survive the numpy → blob → numpy round trip."""
        original = np.random.default_rng(42).standard_normal(1024).astype(np.float32)
        blob = vector_to_blob(original)
        restored = blob_to_vector(blob, 1024)
        np.testing.assert_array_equal(original, restored)

    def test_cosine_similarity_normalized(self):
        """Cosine sim of normalized vectors is just the dot product."""
        rng = np.random.default_rng(42)
        a = rng.standard_normal(1024).astype(np.float32)
        b = rng.standard_normal(1024).astype(np.float32)
        a /= np.linalg.norm(a)
        b /= np.linalg.norm(b)

        sim = cosine_similarity(a, b)
        assert -1.0 <= sim <= 1.0

    def test_identical_vectors_perfect_similarity(self):
        vec = np.random.default_rng(42).standard_normal(1024).astype(np.float32)
        vec /= np.linalg.norm(vec)
        assert cosine_similarity(vec, vec) == pytest.approx(1.0, abs=1e-5)
