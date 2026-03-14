"""Tests for the experiment runner.

Uses mock LLM client and synthetic embeddings to test the full pipeline
without external API calls or model downloads.
"""

import json
import sqlite3
from unittest.mock import patch

import numpy as np
import pytest

from src.embeddings.embedder import store_skill_embedding
from src.experiment.runner import (
    EpisodeResult,
    ExperimentRunner,
    LLMResponse,
)

# ── Test Config ──

TEST_CONFIG = {
    "experiment": {
        "name": "test",
        "seed": 42,
        "tasks_per_condition": 5,
        "total_conditions": 4,
    },
    "embeddings": {
        "model": "Qwen/Qwen3-Embedding-0.6B",
        "dimensions": 1024,
    },
    "search": {
        "bm25_weight": 0.4,
        "dense_weight": 0.6,
        "rrf_k": 60,
        "top_k": 10,
    },
    "weight_presets": {
        "balanced": [0.330, 0.330, 0.340],
        "recency_heavy": [0.600, 0.200, 0.200],
        "importance_heavy": [0.150, 0.700, 0.150],
        "relevance_heavy": [0.200, 0.150, 0.650],
        "relevance_importance": [0.150, 0.350, 0.500],
    },
    "bandit": {
        "prior_alpha": 1.0,
        "prior_beta": 1.0,
    },
    "llm": {
        "provider": "minimax",
        "model": "MiniMax-M2.5",
        "temperature": 0.7,
        "max_tokens": 4096,
    },
    "conditions": {
        1: {
            "name": "control",
            "agent_name": "control_a",
            "use_bandit": False,
            "use_feedback": False,
            "embed_explanations": False,
            "fixed_preset": "balanced",
        },
        2: {
            "name": "dimension_feedback",
            "agent_name": "feedback_a",
            "use_bandit": True,
            "use_feedback": True,
            "embed_explanations": False,
        },
        3: {
            "name": "full_system",
            "agent_name": "full_a",
            "use_bandit": True,
            "use_feedback": True,
            "embed_explanations": True,
        },
        4: {
            "name": "qualitative",
            "agent_name": "qual_a",
            "use_bandit": True,
            "use_feedback": True,
            "embed_explanations": False,
        },
    },
    "runner": {
        "max_steps": 5,  # small for tests
        "synthesis_nudge_pct": 0.6,
        "top_k_skills": 3,
        "recency_decay_window": 50,
        "retry_max": 2,
        "retry_backoff": [0, 0],  # no waiting in tests
    },
    "rate_control": {
        "iterations_per_hour": 1000,  # effectively no limit for tests
        "window_seconds": 3600,
    },
}


class MockLLMClient:
    """Mock LLM that returns predictable responses with feedback blocks."""

    def __init__(self, include_feedback: bool = True):
        self.include_feedback = include_feedback
        self.call_count = 0

    def complete(
        self, prompt: str, temperature: float = 0.7, max_tokens: int = 4096
    ) -> LLMResponse:
        self.call_count += 1

        response_text = (
            "Here is my analysis of the task. "
            "I used the retrieved skills to formulate a comprehensive response "
            "about the topic at hand. The gradient descent algorithm works by "
            "iteratively adjusting parameters in the direction of steepest descent. "
            "This is a well-established optimization technique.\n"
        )

        if self.include_feedback:
            response_text += """
[RETRIEVAL EFFECTIVENESS]
Recency: 3
Importance: 4
Relevance: 5

Brief explanation (2-3 sentences):
- Relevance was most valuable because the topic matched perfectly.
- Recency was least valuable because this is a foundational topic.
- More implementation-focused skills would have been helpful.
[/RETRIEVAL EFFECTIVENESS]"""

        return LLMResponse(
            text=response_text,
            tokens_used=150,
            is_final=True,
            model="test-model",
        )


class MockEmbedder:
    """Mock embedder that returns deterministic random vectors."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def embed(self, text: str) -> np.ndarray:
        vec = self.rng.standard_normal(1024).astype(np.float32)
        vec /= np.linalg.norm(vec)
        return vec


def create_test_db(conn: sqlite3.Connection, rng: np.random.Generator) -> None:
    """Populate a database with test skills and tasks."""
    skills = [
        ("s1", "research", "Gradient Descent", "Optimization via gradient descent"),
        ("s2", "research", "Attention", "Self-attention transformer mechanism"),
        ("s3", "synthesis", "ReAct Pattern", "Reasoning and acting agent loop"),
    ]
    for sid, domain, title, content in skills:
        conn.execute(
            "INSERT INTO skills (skill_id, domain, title, content) VALUES (?, ?, ?, ?)",
            (sid, domain, title, content),
        )
        vec = rng.standard_normal(1024).astype(np.float32)
        vec /= np.linalg.norm(vec)
        store_skill_embedding(conn, sid, vec)

    tasks = [
        ("t1", "ML Fundamentals", "Explain SGD", "Explain stochastic gradient descent"),
        ("t2", "ML Fundamentals", "Attention Mechanisms", "Explain attention mechanisms"),
    ]
    for tid, theme, title, desc in tasks:
        conn.execute(
            "INSERT INTO tasks (task_id, theme, title, description) VALUES (?, ?, ?, ?)",
            (tid, theme, title, desc),
        )

    conn.commit()


@pytest.fixture
def runner():
    """ExperimentRunner with in-memory DB, mock LLM, mock embedder."""
    with patch("src.experiment.runner.load_config", return_value=TEST_CONFIG):
        r = ExperimentRunner(
            config_path="test.yaml",
            db_path=":memory:",
            seed=42,
        )

    # Populate DB
    rng = np.random.default_rng(42)
    create_test_db(r.conn, rng)

    # Set up mock components
    r.llm_client = MockLLMClient(include_feedback=True)
    r.embedder = MockEmbedder(seed=42)

    # Initialize (anchor embeddings etc.)
    r.initialize()

    yield r
    r.conn.close()


@pytest.fixture
def runner_no_feedback():
    """Runner with LLM that doesn't include feedback blocks."""
    with patch("src.experiment.runner.load_config", return_value=TEST_CONFIG):
        r = ExperimentRunner(
            config_path="test.yaml",
            db_path=":memory:",
            seed=42,
        )

    rng = np.random.default_rng(42)
    create_test_db(r.conn, rng)
    r.llm_client = MockLLMClient(include_feedback=False)
    r.embedder = MockEmbedder(seed=42)
    r.initialize()

    yield r
    r.conn.close()


# ── Condition Config ──


class TestConditionConfig:
    def test_builds_all_conditions(self, runner):
        assert len(runner.conditions) == 4

    def test_control_condition(self, runner):
        control = runner.conditions[0]
        assert control.condition_id == 1
        assert control.name == "control"
        assert not control.use_bandit
        assert not control.use_feedback
        assert control.fixed_preset == "balanced"

    def test_ts_conditions_have_bandits(self, runner):
        for cond in runner.conditions:
            if cond.use_bandit:
                assert cond.condition_id in runner.bandits


# ── Weight Selection ──


class TestWeightSelection:
    def test_control_gets_fixed_weights(self, runner):
        control = runner.conditions[0]
        preset_id, weights = runner._select_weights(control)
        assert preset_id == "balanced"
        assert abs(sum(weights) - 1.0) < 0.02

    def test_ts_conditions_use_bandit(self, runner):
        cond2 = runner.conditions[1]
        preset_id, weights = runner._select_weights(cond2)
        assert preset_id in TEST_CONFIG["weight_presets"]
        assert abs(sum(weights) - 1.0) < 0.02


# ── Prompt Building ──


class TestPromptBuilding:
    def test_prompt_contains_task(self, runner):
        task = {"task_id": "t1", "description": "Explain SGD"}
        prompt = runner._build_prompt(task, [], runner.conditions[0])
        assert "Explain SGD" in prompt

    def test_feedback_conditions_get_rubric(self, runner):
        task = {"task_id": "t1", "description": "Explain SGD"}

        # Condition 2 should get Likert rubric
        prompt2 = runner._build_prompt(task, [], runner.conditions[1])
        assert "RETRIEVAL EFFECTIVENESS" in prompt2
        assert "1-5" in prompt2

        # Condition 4 should get qualitative rubric with separated sections
        prompt4 = runner._build_prompt(task, [], runner.conditions[3])
        assert "RETRIEVAL EFFECTIVENESS" in prompt4
        assert "RECENCY EVALUATION" in prompt4
        assert "IMPORTANCE EVALUATION" in prompt4
        assert "RELEVANCE EVALUATION" in prompt4

    def test_control_gets_no_rubric(self, runner):
        task = {"task_id": "t1", "description": "Explain SGD"}
        prompt = runner._build_prompt(task, [], runner.conditions[0])
        assert "RETRIEVAL EFFECTIVENESS" not in prompt


# ── Task Execution ──


class TestTaskExecution:
    def test_execute_returns_response(self, runner):
        text, steps, tokens, inp, out, step_records = runner._execute_task("Test prompt")
        assert len(text) > 0
        assert steps >= 1
        assert tokens > 0
        assert len(step_records) == steps

    def test_execute_respects_step_budget(self, runner):
        """Mock LLM always returns is_final=True, so should complete in 1 step."""
        _, steps, _, _, _, step_records = runner._execute_task("Test prompt")
        assert steps == 1  # is_final=True means 1 step
        assert step_records[0].is_final

    def test_execute_raises_without_llm_client(self, runner):
        runner.llm_client = None
        with pytest.raises(RuntimeError, match="LLM client not set"):
            runner._execute_task("Test prompt")


# ── LLM Retry ──


class TestLLMRetry:
    def test_retry_on_failure(self, runner):
        call_count = 0

        class FailThenSucceedLLM:
            def complete(self, prompt, temperature=0.7, max_tokens=4096):
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    raise ConnectionError("API timeout")
                return LLMResponse(text="Success", tokens_used=50, is_final=True)

        runner.llm_client = FailThenSucceedLLM()
        response = runner._call_llm_with_retry("test")
        assert response.text == "Success"
        assert call_count == 2

    def test_raises_after_max_retries(self, runner):
        class AlwaysFailLLM:
            def complete(self, prompt, temperature=0.7, max_tokens=4096):
                raise ConnectionError("API down")

        runner.llm_client = AlwaysFailLLM()
        with pytest.raises(RuntimeError, match="failed after"):
            runner._call_llm_with_retry("test")


# ── Episode Execution ──


class TestRunEpisode:
    def test_control_episode(self, runner):
        task = {"task_id": "t1", "theme": "ML", "title": "SGD", "description": "Explain SGD"}
        cond = runner.conditions[0]  # control

        result = runner.run_episode(task, cond, task_order=1)
        assert isinstance(result, EpisodeResult)
        assert result.condition_id == 1
        assert result.preset_id == "balanced"
        assert result.step_count >= 1
        assert not result.feedback_parsed  # control has no feedback
        assert result.composite_reward is None

    def test_feedback_episode(self, runner):
        task = {"task_id": "t1", "theme": "ML", "title": "SGD", "description": "Explain SGD"}
        cond = runner.conditions[1]  # dimension_feedback

        result = runner.run_episode(task, cond, task_order=1)
        assert result.condition_id == 2
        assert result.feedback_parsed
        assert result.composite_reward is not None
        assert 0.0 <= result.composite_reward <= 1.0

    def test_episode_records_to_db(self, runner):
        task = {"task_id": "t1", "theme": "ML", "title": "SGD", "description": "Explain SGD"}
        cond = runner.conditions[1]

        result = runner.run_episode(task, cond, task_order=1)

        # Episode recorded
        ep = runner.conn.execute(
            "SELECT * FROM episodes WHERE episode_id = ?", (result.episode_id,)
        ).fetchone()
        assert ep is not None
        assert ep["condition_id"] == 2
        assert ep["step_count"] >= 1
        assert ep["total_tokens"] > 0

    def test_feedback_stored_in_db(self, runner):
        task = {"task_id": "t1", "theme": "ML", "title": "SGD", "description": "Explain SGD"}
        cond = runner.conditions[1]

        result = runner.run_episode(task, cond, task_order=1)

        fb = runner.conn.execute(
            "SELECT * FROM feedback WHERE episode_id = ?", (result.episode_id,)
        ).fetchone()
        assert fb is not None
        assert fb["rating_relevance"] == 5  # from mock LLM's response

    def test_bandit_updated_after_feedback(self, runner):
        task = {"task_id": "t1", "theme": "ML", "title": "SGD", "description": "Explain SGD"}
        cond = runner.conditions[1]

        # Get initial state
        bandit = runner.bandits[cond.condition_id]
        initial_pulls = bandit.get_summary()["total_pulls"]

        result = runner.run_episode(task, cond, task_order=1)

        # Bandit should have been updated
        if result.feedback_parsed:
            assert bandit.get_summary()["total_pulls"] == initial_pulls + 1

    def test_no_feedback_no_bandit_update(self, runner_no_feedback):
        """When feedback fails to parse, bandit should NOT be updated."""
        task = {"task_id": "t1", "theme": "ML", "title": "SGD", "description": "Explain SGD"}
        cond = runner_no_feedback.conditions[1]

        bandit = runner_no_feedback.bandits[cond.condition_id]
        initial_pulls = bandit.get_summary()["total_pulls"]

        result = runner_no_feedback.run_episode(task, cond, task_order=1)

        assert not result.feedback_parsed
        assert bandit.get_summary()["total_pulls"] == initial_pulls

    def test_retrieval_results_recorded(self, runner):
        task = {"task_id": "t1", "theme": "ML", "title": "SGD", "description": "Explain SGD"}
        cond = runner.conditions[1]

        result = runner.run_episode(task, cond, task_order=1)

        rr = runner.conn.execute(
            "SELECT COUNT(*) as n FROM retrieval_results WHERE episode_id = ?",
            (result.episode_id,),
        ).fetchone()
        assert rr["n"] > 0

    def test_skill_usage_recorded(self, runner):
        task = {"task_id": "t1", "theme": "ML", "title": "SGD", "description": "Explain SGD"}
        cond = runner.conditions[1]

        runner.run_episode(task, cond, task_order=1)

        su = runner.conn.execute(
            "SELECT COUNT(*) as n FROM skill_usage WHERE condition_id = ?",
            (cond.condition_id,),
        ).fetchone()
        assert su["n"] > 0


# ── Resumption ──


class TestResumption:
    def test_default_resume_point_is_zero(self, runner):
        assert runner.get_resume_point() == 0

    def test_set_and_get_resume_point(self, runner):
        runner._set_resume_point(42)
        assert runner.get_resume_point() == 42

    def test_resume_point_persists(self, runner):
        runner._set_resume_point(10)
        runner._set_resume_point(20)
        assert runner.get_resume_point() == 20


# ── Initialization ──


class TestInitialization:
    def test_metadata_stored(self, runner):
        row = runner.conn.execute(
            "SELECT value FROM experiment_metadata WHERE key = 'seed'"
        ).fetchone()
        assert row is not None
        assert row["value"] == "42"

    def test_config_stored(self, runner):
        row = runner.conn.execute(
            "SELECT value FROM experiment_metadata WHERE key = 'config'"
        ).fetchone()
        assert row is not None
        config = json.loads(row["value"])
        assert config["experiment"]["name"] == "test"

    def test_anchor_parser_initialized(self, runner):
        assert runner.anchor_parser._initialized

    def test_all_presets_seeded(self, runner):
        rows = runner.conn.execute("SELECT COUNT(*) as n FROM weight_presets").fetchone()
        assert rows["n"] == 5


# ── Idempotency ──


class TestIdempotency:
    def test_duplicate_episode_rejected(self, runner):
        task = {"task_id": "t1", "theme": "ML", "title": "SGD", "description": "Explain SGD"}
        cond = runner.conditions[0]

        # First episode should succeed
        runner.run_episode(task, cond, task_order=1)

        # Second episode with same (condition_id, task_id) caught by prepare phase
        with pytest.raises(RuntimeError, match="already completed"):
            runner.run_episode(task, cond, task_order=1)


# ── Skill Usage Tracking ──


class TestSkillUsageTracking:
    """Verify that skill usage is recorded after episodes."""

    def test_skill_usage_recorded_after_episode(self, runner):
        """After running an episode, skill_usage table should have records."""
        cond = runner.conditions[0]  # control

        task1 = {"task_id": "t1", "theme": "ML", "title": "SGD", "description": "Explain SGD"}
        runner.run_episode(task1, cond, task_order=1)

        usage = runner.conn.execute(
            "SELECT COUNT(DISTINCT skill_id) as n FROM skill_usage WHERE condition_id = ?",
            (cond.condition_id,),
        ).fetchone()
        assert usage["n"] > 0, "skill_usage should be recorded after first episode"

    def test_search_results_have_relevance_scores(self, runner):
        """All search results should have valid relevance_score from dense search."""
        cond = runner.conditions[1]  # dimension_feedback

        task1 = {"task_id": "t1", "theme": "ML", "title": "SGD", "description": "Explain SGD"}
        runner.run_episode(task1, cond, task_order=1)

        task2 = {
            "task_id": "t2",
            "theme": "ML",
            "title": "Attention",
            "description": "Explain attention mechanisms",
            "ground_truth_skills": "[]",
        }
        embedding = runner.embedder.embed(task2["description"])
        ctx = runner._prepare_episode(task2, cond, task_order=2, task_embedding=embedding)
        assert ctx is not None

        for r in ctx.search_results:
            assert isinstance(r.relevance_score, float)
