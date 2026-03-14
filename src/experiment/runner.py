"""
Experiment runner: orchestrates 4 conditions × 300 tasks = 1,200 iterations.

Architecture (SYSTEM_DESIGN.md §9):
  For each task (round-robin across conditions):
    1. SELECT WEIGHTS  — Control: fixed. Others: Thompson Sampling.
    2. RETRIEVE SKILLS  — Stage 1 (BM25+dense+RRF) → Stage 2 (apply weights)
    3. BUILD PROMPT     — Inject skills + append rubric (Conditions 2-4)
    4. EXECUTE          — MiniMax M2.5 API, up to max_steps turns
    5. PARSE FEEDBACK   — Condition 1: skip. 2-3: Likert regex. 4: anchor embedding.
    6. UPDATE BANDIT    — Composite reward → Beta posterior update
    7. LOG EVERYTHING   — Full reproducibility: prompt, response, ratings, posterior

Supports clean resumption via experiment_metadata['last_completed_task'].
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from src.db.schema import init_bandit_state, init_db, log_event, seed_weight_presets
from src.embeddings.embedder import Embedder, store_feedback_embedding
from src.experiment.bandit import ThompsonSamplingBandit
from src.experiment.feedback_parser import (
    RUBRIC_LIKERT,
    RUBRIC_QUALITATIVE,
    SeparatedAnchorParser,
    extract_qualitative_sections,
    parse_likert_feedback,
    parse_qualitative_feedback,
)
from src.experiment.rate_control import SYNTHESIS_NUDGE, SlidingWindowLimiter, StepBudget
from src.experiment.scoring import record_skill_usage, score_search_results
from src.observability.tracing import EpisodeTrace, ExperimentTracer
from src.search.hybrid import HybridSearchEngine
from src.utils.config import load_config

logger = logging.getLogger(__name__)


# ── LLM Client Protocol ──


class LLMClient(Protocol):
    """Abstract interface for the LLM backend. Swap MiniMax for mock in tests."""

    def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> "LLMResponse": ...


@dataclass
class LLMResponse:
    """Response from an LLM call."""

    text: str
    tokens_used: int = 0  # input + output tokens
    input_tokens: int = 0
    output_tokens: int = 0
    is_final: bool = True  # for multi-turn: True if agent is done
    model: str = ""


# ── Condition Runner ──


@dataclass
class ConditionConfig:
    """Configuration for one experimental condition."""

    condition_id: int
    name: str
    agent_name: str
    use_bandit: bool
    use_feedback: bool
    embed_explanations: bool
    fixed_preset: str | None = None  # Only for control


@dataclass
class EpisodeResult:
    """Result of running one task under one condition."""

    episode_id: int
    condition_id: int
    task_id: str
    preset_id: str
    success: bool
    step_count: int
    total_tokens: int
    input_tokens: int
    output_tokens: int
    feedback_parsed: bool
    composite_reward: float | None


@dataclass
class _StepRecord:
    """One LLM API call within an episode."""

    step_number: int
    prompt_text: str
    response_text: str
    input_tokens: int
    output_tokens: int
    is_final: bool


@dataclass
class _EpisodeContext:
    """Intermediate state passed between prepare → execute → finalize phases."""

    task: dict
    cond: ConditionConfig
    task_order: int
    preset_id: str
    weights: tuple
    search_results: list
    prompt: str
    trace: EpisodeTrace | None = None
    # Filled after LLM execution (phase 2)
    response_text: str = ""
    step_count: int = 0
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    step_records: list = field(default_factory=list)
    start_time_ms: float = 0.0
    end_time_ms: float = 0.0


# ── Main Runner ──


class ExperimentRunner:
    """Orchestrates the full experiment across all conditions and tasks."""

    def __init__(
        self,
        config_path: str | Path = "configs/experiment.yaml",
        db_path: str | Path = "experiment_v3.db",
        llm_client: LLMClient | None = None,
        embedder: Embedder | None = None,
        seed: int = 42,
    ):
        self.config = load_config(config_path)
        self.db_path = db_path
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Initialize DB
        self.conn = init_db(db_path)

        # Seed weight presets from config
        seed_weight_presets(self.conn, self.config["weight_presets"])

        # Initialize components
        self.llm_client = llm_client
        self.embedder = embedder
        self.search_engine = HybridSearchEngine(
            conn=self.conn,
            rrf_k=self.config["search"]["rrf_k"],
            bm25_weight=self.config["search"]["bm25_weight"],
            dense_weight=self.config["search"]["dense_weight"],
        )

        # Rate control
        rc = self.config.get("rate_control", {})
        self.rate_limiter = SlidingWindowLimiter(
            max_per_window=rc.get("iterations_per_hour", 50),
            window_seconds=rc.get("window_seconds", 3600),
        )

        runner_cfg = self.config.get("runner", {})
        self.max_steps = runner_cfg.get("max_steps", 35)
        self.top_k_skills = runner_cfg.get("top_k_skills", 5)
        self.recency_decay_window = runner_cfg.get("recency_decay_window", 50)
        self.retry_max = runner_cfg.get("retry_max", 3)
        self.retry_backoff = runner_cfg.get("retry_backoff", [2, 4, 8])

        # Separated-question anchor parser for Condition 4
        self.anchor_parser = SeparatedAnchorParser()

        # Langfuse tracing (reads keys from env; graceful no-op if missing)
        self.tracer = ExperimentTracer()

        # Build condition configs
        self.conditions = self._build_conditions()

        # Bandits (one per TS condition)
        self.bandits: dict[int, ThompsonSamplingBandit] = {}
        preset_ids = list(self.config["weight_presets"].keys())
        for cond in self.conditions:
            if cond.use_bandit:
                init_bandit_state(
                    self.conn,
                    [cond.condition_id],
                    preset_ids,
                    prior_alpha=self.config["bandit"]["prior_alpha"],
                    prior_beta=self.config["bandit"]["prior_beta"],
                )
                discount = self.config["bandit"].get("discount")
                self.bandits[cond.condition_id] = ThompsonSamplingBandit(
                    self.conn,
                    cond.condition_id,
                    rng=np.random.default_rng(self.seed + cond.condition_id),
                    discount=discount,
                )

    def _build_conditions(self) -> list[ConditionConfig]:
        """Build ConditionConfig objects from YAML."""
        conditions = []
        for cid, cdata in self.config["conditions"].items():
            conditions.append(
                ConditionConfig(
                    condition_id=int(cid),
                    name=cdata["name"],
                    agent_name=cdata.get("agent_name", f"agent_{cid}"),
                    use_bandit=cdata["use_bandit"],
                    use_feedback=cdata["use_feedback"],
                    embed_explanations=cdata.get("embed_explanations", False),
                    fixed_preset=cdata.get("fixed_preset"),
                )
            )
        return sorted(conditions, key=lambda c: c.condition_id)

    def initialize(self) -> None:
        """One-time initialization before experiment starts.

        Call this after setting llm_client and embedder.
        """
        if self.embedder is not None:
            self.anchor_parser.initialize(self.embedder)

        # Store experiment metadata
        metadata = {
            "seed": str(self.seed),
            "config": json.dumps(self.config),
            "start_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "embedding_model": self.config["embeddings"]["model"],
            "llm_model": self.config["llm"]["model"],
        }
        for key, value in metadata.items():
            self.conn.execute(
                "INSERT OR REPLACE INTO experiment_metadata (key, value) VALUES (?, ?)",
                (key, value),
            )
        self.conn.commit()
        log_event(self.conn, "runner", "Experiment initialized", metadata=json.dumps(metadata))

    def get_resume_point(self) -> int:
        """Get the last completed task order for resumption."""
        row = self.conn.execute(
            "SELECT value FROM experiment_metadata WHERE key = 'last_completed_task'"
        ).fetchone()
        return int(row["value"]) if row else 0

    def _set_resume_point(self, task_order: int) -> None:
        """Update the resumption checkpoint."""
        self.conn.execute(
            "INSERT OR REPLACE INTO experiment_metadata (key, value) VALUES (?, ?)",
            ("last_completed_task", str(task_order)),
        )
        self.conn.commit()

    def load_tasks(self) -> list[dict]:
        """Load all tasks ordered by task_order."""
        rows = self.conn.execute(
            "SELECT task_id, theme, title, description,"
            " difficulty, ground_truth_skills"
            " FROM tasks ORDER BY rowid"
        ).fetchall()
        return [dict(row) for row in rows]

    def _select_weights(self, cond: ConditionConfig) -> tuple[str, tuple[float, float, float]]:
        """Select weight preset for this condition.

        Returns (preset_id, (w_recency, w_importance, w_relevance)).
        """
        if not cond.use_bandit:
            # Control: fixed preset
            preset_id = cond.fixed_preset or "balanced"
        else:
            preset_id = self.bandits[cond.condition_id].select_arm()

        row = self.conn.execute(
            "SELECT w_recency, w_importance, w_relevance FROM weight_presets WHERE preset_id = ?",
            (preset_id,),
        ).fetchone()

        return preset_id, (row["w_recency"], row["w_importance"], row["w_relevance"])

    def _build_prompt(
        self,
        task: dict,
        skills: list,
        cond: ConditionConfig,
    ) -> str:
        """Build the full prompt with skills and rubric injection."""
        parts = []

        # Task description
        parts.append(f"## Task\n{task['description']}\n")

        # Inject retrieved skills
        if skills:
            parts.append("## Retrieved Skills\n")
            for i, result in enumerate(skills, 1):
                skill_row = self.conn.execute(
                    "SELECT title, content FROM skills WHERE skill_id = ?",
                    (result.skill_id,),
                ).fetchone()
                if skill_row:
                    parts.append(f"### Skill {i}: {skill_row['title']}")
                    parts.append(skill_row["content"])
                    parts.append("")

        # Append rubric (Conditions 2-4 only)
        if cond.use_feedback:
            if cond.condition_id == 4:
                parts.append(RUBRIC_QUALITATIVE)
            else:
                parts.append(RUBRIC_LIKERT)

        return "\n".join(parts)

    def _execute_task(
        self,
        prompt: str,
        trace=None,
    ) -> tuple[str, int, int, int, int, list[_StepRecord]]:
        """Execute a task via LLM with step budget.

        Returns (response_text, step_count, total_tokens, input_tokens,
                 output_tokens, step_records).
        """
        if self.llm_client is None:
            raise RuntimeError("LLM client not set. Call runner.llm_client = ... first.")

        budget = StepBudget(max_steps=self.max_steps)
        conversation_parts = [prompt]
        total_tokens = 0
        input_tokens = 0
        output_tokens = 0
        step_records: list[_StepRecord] = []

        for _ in range(self.max_steps):
            exhausted = budget.step()

            # Metacognitive guardrail
            has_output = len(conversation_parts) > 1
            if budget.should_nudge(has_output):
                conversation_parts.append(SYNTHESIS_NUDGE)

            # API call with retry
            full_prompt = "\n".join(conversation_parts)
            response = self._call_llm_with_retry(full_prompt)
            total_tokens += response.tokens_used
            input_tokens += response.input_tokens
            output_tokens += response.output_tokens
            conversation_parts.append(response.text)

            done = response.is_final or exhausted
            step_records.append(
                _StepRecord(
                    step_number=budget.current_step,
                    prompt_text=full_prompt,
                    response_text=response.text,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    is_final=done,
                )
            )

            # Langfuse: log each LLM call
            if trace is not None:
                trace.log_llm_call(
                    prompt=full_prompt,
                    response_text=response.text,
                    tokens_used=response.tokens_used,
                    model=response.model,
                    step=budget.current_step,
                )

            if done:
                break

        final_response = conversation_parts[-1]
        return (
            final_response,
            budget.current_step,
            total_tokens,
            input_tokens,
            output_tokens,
            step_records,
        )

    def _call_llm_with_retry(self, prompt: str) -> LLMResponse:
        """Call LLM with exponential backoff retry."""
        assert self.llm_client is not None, "LLM client not set"
        last_error = None
        for attempt in range(self.retry_max):
            try:
                return self.llm_client.complete(
                    prompt=prompt,
                    temperature=self.config["llm"]["temperature"],
                    max_tokens=self.config["llm"]["max_tokens"],
                )
            except Exception as e:
                last_error = e
                if attempt < self.retry_max - 1:
                    wait = self.retry_backoff[min(attempt, len(self.retry_backoff) - 1)]
                    logger.warning(
                        f"LLM call failed (attempt {attempt + 1}), retrying in {wait}s: {e}"
                    )
                    time.sleep(wait)

        raise RuntimeError(f"LLM call failed after {self.retry_max} retries: {last_error}")

    def _prepare_episode(
        self,
        task: dict,
        cond: ConditionConfig,
        task_order: int,
        task_embedding=None,
    ) -> _EpisodeContext | None:
        """Phase 1: Select weights, retrieve skills, build prompt. Runs on main thread."""

        # Idempotency check
        existing = self.conn.execute(
            "SELECT episode_id FROM episodes WHERE condition_id = ? AND task_id = ?",
            (cond.condition_id, task["task_id"]),
        ).fetchone()
        if existing:
            logger.info(f"  Condition {cond.name}: already completed, skipping")
            return None

        # 1. SELECT WEIGHTS
        preset_id, weights = self._select_weights(cond)

        # Start Langfuse trace
        trace = self.tracer.start_episode(
            condition_name=cond.name,
            condition_id=cond.condition_id,
            task_id=task["task_id"],
            task_order=task_order,
            preset_id=preset_id,
            task_title=task.get("title", ""),
        )

        # 2. RETRIEVE SKILLS — RRF candidates only
        # All conditions get the same candidate pool from hybrid search (Pool A).
        # Weight presets differentiate by scoring the same candidates differently.
        # Pool B (usage-history injection) removed: it caused lock-in where
        # previously-used skills with high recency/importance were mathematically
        # unbeatable, making C1 degenerate (same 5 skills for all tasks).
        search_results = []
        if task_embedding is not None:
            raw_results = self.search_engine.search(
                query=task["description"],
                query_embedding=task_embedding,
                top_k=self.top_k_skills * 3,
            )

            score_search_results(
                self.conn,
                cond.condition_id,
                task_order,
                raw_results,
                self.recency_decay_window,
            )
            for r in raw_results:
                r.final_score = (
                    weights[0] * r.recency_score
                    + weights[1] * r.importance_score
                    + weights[2] * r.relevance_score
                )
            raw_results.sort(key=lambda x: x.final_score, reverse=True)
            search_results = raw_results[: self.top_k_skills]

        # 3. BUILD PROMPT
        prompt = self._build_prompt(task, search_results, cond)

        return _EpisodeContext(
            task=task,
            cond=cond,
            task_order=task_order,
            preset_id=preset_id,
            weights=weights,
            search_results=search_results,
            prompt=prompt,
            trace=trace,
        )

    def _finalize_episode(self, ctx: _EpisodeContext) -> EpisodeResult:
        """Phase 3: Parse feedback, write DB, update bandit. Runs on main thread."""

        success = len(ctx.response_text) > 100

        # Record episode
        duration_ms = int(ctx.end_time_ms - ctx.start_time_ms) if ctx.end_time_ms else None
        self.conn.execute(
            """INSERT INTO episodes
               (condition_id, task_id, preset_id, task_order,
                prompt_sent, llm_response, success, step_count, total_tokens,
                input_tokens, output_tokens, duration_ms, completed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
            (
                ctx.cond.condition_id,
                ctx.task["task_id"],
                ctx.preset_id,
                ctx.task_order,
                ctx.prompt,
                ctx.response_text,
                int(success),
                ctx.step_count,
                ctx.total_tokens,
                ctx.input_tokens,
                ctx.output_tokens,
                duration_ms,
            ),
        )
        self.conn.commit()
        episode_id = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        # Record retrieval results with dimension scores
        gt_skills = json.loads(ctx.task.get("ground_truth_skills", "[]") or "[]")
        for rank, r in enumerate(ctx.search_results):
            self.conn.execute(
                """INSERT INTO retrieval_results
                   (episode_id, skill_id, rank, bm25_score, dense_score, rrf_score,
                    recency_score, importance_score, relevance_score, final_score,
                    is_ground_truth)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    episode_id,
                    r.skill_id,
                    rank,
                    r.bm25_score,
                    r.dense_score,
                    r.rrf_score,
                    r.recency_score,
                    r.importance_score,
                    r.relevance_score,
                    r.final_score,
                    int(r.skill_id in gt_skills),
                ),
            )

        # Record step-level API call log
        for step in ctx.step_records:
            self.conn.execute(
                """INSERT INTO step_log
                   (episode_id, step_number, prompt_text, response_text,
                    input_tokens, output_tokens, is_final)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    episode_id,
                    step.step_number,
                    step.prompt_text,
                    step.response_text,
                    step.input_tokens,
                    step.output_tokens,
                    int(step.is_final),
                ),
            )

        # Record skill usage
        for r in ctx.search_results:
            record_skill_usage(
                self.conn,
                ctx.cond.condition_id,
                r.skill_id,
                episode_id,
                int(success),
                ctx.task_order,
            )
        self.conn.commit()

        # PARSE FEEDBACK
        feedback_parsed = False
        composite_reward = None
        parsed = None

        if ctx.cond.use_feedback:
            if ctx.cond.condition_id == 4:
                sections = extract_qualitative_sections(ctx.response_text)
                if sections is not None and self.embedder is not None:
                    section_embeddings = {
                        dim: self.embedder.embed(text) for dim, text in sections.items()
                    }
                    parsed = parse_qualitative_feedback(
                        ctx.response_text,
                        section_embeddings,
                        self.anchor_parser,
                    )
            else:
                parsed = parse_likert_feedback(ctx.response_text)

            if parsed is not None:
                feedback_parsed = True
                composite_reward = parsed.composite_reward

                self.conn.execute(
                    """INSERT INTO feedback
                       (episode_id, rating_recency, rating_importance, rating_relevance,
                        explanation, inferred_recency, inferred_importance, inferred_relevance)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        episode_id,
                        int(parsed.recency * 4 + 1) if parsed.parse_method == "likert" else None,
                        int(parsed.importance * 4 + 1) if parsed.parse_method == "likert" else None,
                        int(parsed.relevance * 4 + 1) if parsed.parse_method == "likert" else None,
                        parsed.explanation,
                        parsed.recency if parsed.parse_method == "separated_anchor" else None,
                        parsed.importance if parsed.parse_method == "separated_anchor" else None,
                        parsed.relevance if parsed.parse_method == "separated_anchor" else None,
                    ),
                )
                self.conn.commit()
                feedback_id = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]

                if ctx.cond.embed_explanations and self.embedder is not None:
                    explanation_embedding = self.embedder.embed(parsed.explanation)
                    store_feedback_embedding(self.conn, feedback_id, explanation_embedding)

                if ctx.cond.use_bandit:
                    self.bandits[ctx.cond.condition_id].update(ctx.preset_id, composite_reward)

                if ctx.cond.use_bandit:
                    summary = self.bandits[ctx.cond.condition_id].get_summary()
                    log_event(
                        self.conn,
                        "bandit",
                        f"Condition {ctx.cond.condition_id} after task {ctx.task_order}",
                        metadata=json.dumps(summary),
                    )

        # Langfuse
        if ctx.trace is not None:
            if parsed is not None:
                ctx.trace.log_feedback(
                    parsed=True,
                    reward=composite_reward,
                    method=parsed.parse_method,
                    ratings=parsed.ratings_tuple,
                )
            elif ctx.cond.use_feedback:
                ctx.trace.log_feedback(parsed=False)
            ctx.trace.end(success=success, step_count=ctx.step_count, total_tokens=ctx.total_tokens)

        return EpisodeResult(
            episode_id=episode_id,
            condition_id=ctx.cond.condition_id,
            task_id=ctx.task["task_id"],
            preset_id=ctx.preset_id,
            success=success,
            step_count=ctx.step_count,
            total_tokens=ctx.total_tokens,
            input_tokens=ctx.input_tokens,
            output_tokens=ctx.output_tokens,
            feedback_parsed=feedback_parsed,
            composite_reward=composite_reward,
        )

    def run_episode(
        self,
        task: dict,
        cond: ConditionConfig,
        task_order: int,
    ) -> EpisodeResult:
        """Run one task under one condition. Convenience wrapper for sequential use."""
        task_embedding = None
        if self.embedder is not None:
            task_embedding = self.embedder.embed(task["description"])

        ctx = self._prepare_episode(task, cond, task_order, task_embedding)
        if ctx is None:
            raise RuntimeError(f"Episode already completed: {cond.name} × {task['task_id']}")

        (
            ctx.response_text,
            ctx.step_count,
            ctx.total_tokens,
            ctx.input_tokens,
            ctx.output_tokens,
            ctx.step_records,
        ) = self._execute_task(ctx.prompt, trace=ctx.trace)
        return self._finalize_episode(ctx)

    def run(
        self, start_from: int | None = None, should_stop: Callable[[], bool] | None = None
    ) -> None:
        """Run the full experiment with shuffled tasks and parallel conditions.

        Architecture: Serial → Parallel → Serial (3-phase)
          Phase 1 (serial): Embed task, search, build prompts for all conditions
          Phase 2 (parallel): Fire LLM API calls simultaneously via ThreadPoolExecutor
          Phase 3 (serial): Parse feedback, write DB, update bandits

        This overlaps the LLM API wait (~30-40s) across 4 conditions, yielding
        ~3.5× speedup vs sequential execution.

        Args:
            start_from: Task order to start from. If None, auto-detects from
                        experiment_metadata for clean resumption.
            should_stop: Optional callable returning True to request graceful shutdown.
        """
        tasks = self.load_tasks()
        if not tasks:
            raise RuntimeError("No tasks loaded. Generate task corpus first (step 5).")

        # Deterministic shuffle — breaks sequential-by-theme confound
        # numpy rng.shuffle operates in-place, seed ensures reproducibility
        self.rng.shuffle(tasks)

        # Store shuffle order for reproducibility
        shuffle_order = [t["task_id"] for t in tasks]
        self.conn.execute(
            "INSERT OR REPLACE INTO experiment_metadata (key, value) VALUES (?, ?)",
            ("task_shuffle_order", json.dumps(shuffle_order)),
        )
        self.conn.commit()

        resume_point = start_from if start_from is not None else self.get_resume_point()

        logger.info(f"Starting experiment: {len(tasks)} tasks × {len(self.conditions)} conditions")
        logger.info(f"Resuming from task_order={resume_point} (tasks shuffled, seed={self.seed})")

        for task_order, task in enumerate(tasks, 1):
            if task_order <= resume_point:
                continue

            # Graceful shutdown check
            if should_stop is not None and should_stop():
                logger.info(f"Shutdown requested after task {task_order - 1}. Safe to resume.")
                break

            logger.info(
                f"[Task {task_order}/{len(tasks)}] {task['title']} (theme={task.get('theme', '?')})"
            )

            # ── Phase 1: Prepare (serial) ──
            # Embed task once — shared across all conditions
            task_embedding = None
            if self.embedder is not None:
                task_embedding = self.embedder.embed(task["description"])

            # Build episode contexts for each condition
            contexts: list[_EpisodeContext] = []
            for cond in self.conditions:
                ctx = self._prepare_episode(task, cond, task_order, task_embedding)
                if ctx is not None:
                    contexts.append(ctx)

            if not contexts:
                logger.info("  All conditions already completed, skipping")
                continue

            # Rate limit: reserve slots for all pending conditions
            for _ in contexts:
                check = self.rate_limiter.check()
                if check.action == "throttle":
                    logger.info(f"  Throttled: {check.reason}")
                    time.sleep(check.wait_seconds)

            # ── Phase 2: Execute LLM calls (parallel) ──
            def _execute_for_ctx(ctx: _EpisodeContext) -> _EpisodeContext:
                """Thread target: only does the LLM API call (I/O-bound, GIL-released)."""
                ctx.start_time_ms = time.time() * 1000
                (
                    ctx.response_text,
                    ctx.step_count,
                    ctx.total_tokens,
                    ctx.input_tokens,
                    ctx.output_tokens,
                    ctx.step_records,
                ) = self._execute_task(ctx.prompt, trace=ctx.trace)
                ctx.end_time_ms = time.time() * 1000
                return ctx

            with ThreadPoolExecutor(max_workers=len(contexts)) as executor:
                future_to_ctx = {executor.submit(_execute_for_ctx, ctx): ctx for ctx in contexts}
                completed_contexts: list[_EpisodeContext] = []
                for future in as_completed(future_to_ctx):
                    ctx = future_to_ctx[future]
                    try:
                        completed_ctx = future.result()
                        completed_contexts.append(completed_ctx)
                    except Exception as e:
                        logger.error(f"  Condition {ctx.cond.name}: LLM FAILED - {e}")
                        log_event(
                            self.conn,
                            "runner",
                            f"Episode failed: condition={ctx.cond.condition_id}"
                            f" task={task['task_id']}",
                            level="ERROR",
                            metadata=json.dumps({"error": str(e)}),
                        )

            # ── Phase 3: Finalize (serial) ──
            # Sort by condition_id for deterministic DB write order
            completed_contexts.sort(key=lambda c: c.cond.condition_id)

            for ctx in completed_contexts:
                try:
                    result = self._finalize_episode(ctx)
                    self.rate_limiter.record()

                    status = "✓" if result.success else "✗"
                    feedback = (
                        f"reward={result.composite_reward:.2f}"
                        if result.composite_reward
                        else "no feedback"
                    )
                    logger.info(
                        f"  Condition {ctx.cond.name}: {status} "
                        f"preset={result.preset_id} steps={result.step_count} {feedback}"
                    )
                except Exception as e:
                    logger.error(f"  Condition {ctx.cond.name}: finalize FAILED - {e}")
                    log_event(
                        self.conn,
                        "runner",
                        f"Finalize failed: condition={ctx.cond.condition_id}"
                        f" task={task['task_id']}",
                        level="ERROR",
                        metadata=json.dumps({"error": str(e)}),
                    )

            # Checkpoint after all conditions complete for this task
            self._set_resume_point(task_order)

        logger.info("Experiment complete!")
        self.conn.execute(
            "INSERT OR REPLACE INTO experiment_metadata (key, value) VALUES (?, ?)",
            ("end_timestamp", time.strftime("%Y-%m-%dT%H:%M:%S")),
        )
        self.conn.commit()
        self.tracer.flush()
