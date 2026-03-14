#!/usr/bin/env python3
"""
Integration test: 10 tasks × 4 conditions, end-to-end with real MiniMax API.

Step 12 in the experiment build checklist (STATUS.md).

Validates:
  1. Full pipeline: DB → embed → search → prompt → LLM → parse → bandit → log
  2. Condition 1: Fixed balanced weights, no feedback, no bandit update
  3. Condition 2: TS + Likert parsed, bandit updates, explanations logged only
  4. Condition 3: Same as 2 PLUS explanation embeddings stored (key differentiator)
  5. Condition 4: TS + free-text sections, anchor parser infers ratings
  6. Parse rates ≥ 30% (lenient for n=10; full run expects >65%)
  7. Bandit posteriors diverge from uniform priors
  8. All DB tables populated correctly

Requirements:
  - MINIMAX_API_KEY environment variable
  - ~40 MiniMax API calls
  - Qwen3-Embedding-0.6B model (auto-downloads ~1.2GB on first run)

Usage:
  python scripts/run_integration_test.py [--tasks-per-theme 2] [--keep-db]
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.db.schema import init_db  # noqa: E402
from src.embeddings.embedder import Embedder, store_skill_embedding  # noqa: E402
from src.experiment.runner import ExperimentRunner  # noqa: E402
from src.llm.minimax_client import MiniMaxClient  # noqa: E402

logger = logging.getLogger("integration_test")


# ── Data loading ──


def build_skill_content(skill: dict) -> str:
    """Build searchable content from skill metadata (mirrors load_skills.py)."""
    parts = [
        skill["description"],
        "",
        f"When to use: {skill['when_to_use']}",
        "",
        "Steps:",
    ]
    for step in skill["steps"]:
        parts.append(f"- {step}")
    parts.append("")
    parts.append(f"Expected outcome: {skill['expected_outcome']}")
    return "\n".join(parts)


def select_tasks(corpus_path: Path, n_per_theme: int = 2) -> list[dict]:
    """Select a balanced subset: n_per_theme tasks from each of the 5 themes."""
    with open(corpus_path) as f:
        corpus = json.load(f)

    themes: dict[str, list[dict]] = {}
    for task in corpus:
        themes.setdefault(task["theme"], []).append(task)

    selected = []
    for theme in sorted(themes):
        # Take 1 easy + 1 medium for each theme if n_per_theme=2
        theme_tasks = themes[theme]
        selected.extend(theme_tasks[:n_per_theme])

    logger.info(f"Selected {len(selected)} tasks from {len(themes)} themes")
    return selected


def load_data(conn, tasks: list[dict], skills_path: Path) -> list[dict]:
    """Load tasks and skills into the database. Returns skill list."""
    for task in tasks:
        conn.execute(
            """INSERT OR REPLACE INTO tasks
               (task_id, theme, title, description, difficulty,
                ground_truth_skills, expected_approach)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                task["task_id"],
                task["theme"],
                task["title"],
                task["description"],
                task["difficulty"],
                json.dumps(task.get("ground_truth_skills", [])),
                task.get("expected_approach", ""),
            ),
        )

    with open(skills_path) as f:
        skills = json.load(f)

    for skill in skills:
        content = build_skill_content(skill)
        conn.execute(
            """INSERT OR REPLACE INTO skills (skill_id, domain, title, content)
               VALUES (?, ?, ?, ?)""",
            (skill["skill_id"], skill["domain"], skill["title"], content),
        )
    conn.commit()
    return skills


def embed_skills(conn, embedder: Embedder, skills: list[dict]) -> None:
    """Embed all skills with the real model and store in DB."""
    contents = [build_skill_content(s) for s in skills]
    t0 = time.time()
    embeddings = embedder.embed_batch(contents)
    elapsed = time.time() - t0

    for skill, emb in zip(skills, embeddings):
        store_skill_embedding(conn, skill["skill_id"], emb)

    logger.info(f"Embedded {len(skills)} skills in {elapsed:.1f}s")


# ── Validation ──


def validate(conn, n_tasks: int) -> dict:
    """Run all validation checks against the DB state after the experiment."""
    checks = []

    def check(name, passed, detail=""):
        checks.append({"name": name, "passed": passed, "detail": detail})

    n_conditions = 4
    expected = n_tasks * n_conditions

    # --- Episode counts ---
    total = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
    check("Total episodes", total == expected, f"{total}/{expected}")

    for cid in range(1, 5):
        n = conn.execute("SELECT COUNT(*) FROM episodes WHERE condition_id = ?", (cid,)).fetchone()[
            0
        ]
        check(f"Condition {cid} episode count", n == n_tasks, f"{n}/{n_tasks}")

    # --- Condition 1: control correctness ---
    c1_feedback = conn.execute(
        """SELECT COUNT(*) FROM feedback f
           JOIN episodes e ON f.episode_id = e.episode_id
           WHERE e.condition_id = 1"""
    ).fetchone()[0]
    check("Condition 1: no feedback stored", c1_feedback == 0, f"{c1_feedback}")

    c1_presets = [
        r["preset_id"]
        for r in conn.execute(
            "SELECT DISTINCT preset_id FROM episodes WHERE condition_id = 1"
        ).fetchall()
    ]
    check(
        "Condition 1: always balanced preset", c1_presets == ["balanced"], f"presets={c1_presets}"
    )

    # --- Feedback parse rates (conditions 2-4) ---
    cond_names = {2: "dimension_feedback", 3: "full_system", 4: "qualitative"}
    for cid in [2, 3, 4]:
        total_ep = conn.execute(
            "SELECT COUNT(*) FROM episodes WHERE condition_id = ?", (cid,)
        ).fetchone()[0]
        parsed = conn.execute(
            """SELECT COUNT(*) FROM feedback f
               JOIN episodes e ON f.episode_id = e.episode_id
               WHERE e.condition_id = ?""",
            (cid,),
        ).fetchone()[0]
        rate = parsed / total_ep if total_ep > 0 else 0
        check(
            f"Condition {cid} ({cond_names[cid]}) parse rate ≥ 30%",
            rate >= 0.3,
            f"{parsed}/{total_ep} ({rate:.0%})",
        )

    # --- Correct rating types per condition ---
    # Conditions 2-3: Likert ratings (integer 1-5) should be present
    for cid in [2, 3]:
        likert = conn.execute(
            """SELECT COUNT(*) FROM feedback f
               JOIN episodes e ON f.episode_id = e.episode_id
               WHERE e.condition_id = ?
               AND f.rating_recency IS NOT NULL""",
            (cid,),
        ).fetchone()[0]
        total_fb = conn.execute(
            """SELECT COUNT(*) FROM feedback f
               JOIN episodes e ON f.episode_id = e.episode_id
               WHERE e.condition_id = ?""",
            (cid,),
        ).fetchone()[0]
        check(
            f"Condition {cid}: Likert ratings present",
            likert == total_fb and total_fb > 0,
            f"{likert}/{total_fb}",
        )

    # Condition 4: inferred (float) ratings present
    c4_inferred = conn.execute(
        """SELECT COUNT(*) FROM feedback f
           JOIN episodes e ON f.episode_id = e.episode_id
           WHERE e.condition_id = 4
           AND f.inferred_recency IS NOT NULL"""
    ).fetchone()[0]
    c4_total_fb = conn.execute(
        """SELECT COUNT(*) FROM feedback f
           JOIN episodes e ON f.episode_id = e.episode_id
           WHERE e.condition_id = 4"""
    ).fetchone()[0]
    check(
        "Condition 4: inferred ratings present",
        c4_inferred == c4_total_fb and c4_total_fb > 0,
        f"{c4_inferred}/{c4_total_fb}",
    )

    # --- Condition 3 differentiation: feedback embeddings ---
    c3_feedback = conn.execute(
        """SELECT COUNT(*) FROM feedback f
           JOIN episodes e ON f.episode_id = e.episode_id
           WHERE e.condition_id = 3"""
    ).fetchone()[0]
    c3_embeds = conn.execute(
        """SELECT COUNT(*) FROM feedback_embeddings fe
           JOIN feedback f ON fe.feedback_id = f.feedback_id
           JOIN episodes e ON f.episode_id = e.episode_id
           WHERE e.condition_id = 3"""
    ).fetchone()[0]
    check(
        "Condition 3: all parsed feedback has embeddings",
        c3_embeds == c3_feedback and c3_feedback > 0,
        f"{c3_embeds}/{c3_feedback}",
    )

    # Condition 2: NO feedback embeddings (this is the key differentiator)
    c2_embeds = conn.execute(
        """SELECT COUNT(*) FROM feedback_embeddings fe
           JOIN feedback f ON fe.feedback_id = f.feedback_id
           JOIN episodes e ON f.episode_id = e.episode_id
           WHERE e.condition_id = 2"""
    ).fetchone()[0]
    check("Condition 2: no feedback embeddings (vs Condition 3)", c2_embeds == 0, f"{c2_embeds}")

    # --- Bandit divergence from uniform priors ---
    for cid in [2, 3, 4]:
        arms = conn.execute(
            "SELECT alpha, beta, pulls FROM bandit_state WHERE condition_id = ?",
            (cid,),
        ).fetchall()
        total_pulls = sum(a["pulls"] for a in arms)
        diverged = any(a["alpha"] != 1.0 or a["beta"] != 1.0 for a in arms)
        check(
            f"Condition {cid} bandit posteriors diverged",
            diverged and total_pulls > 0,
            f"total_pulls={total_pulls}",
        )

    # --- Retrieval results populated ---
    rr = conn.execute("SELECT COUNT(*) FROM retrieval_results").fetchone()[0]
    check("Retrieval results populated", rr > 0, f"{rr} rows")

    # --- Skill usage tracked ---
    usage = conn.execute("SELECT COUNT(*) FROM skill_usage").fetchone()[0]
    check("Skill usage tracked", usage > 0, f"{usage} rows")

    # --- All episodes have LLM responses ---
    empty = conn.execute(
        "SELECT COUNT(*) FROM episodes WHERE llm_response IS NULL OR llm_response = ''"
    ).fetchone()[0]
    check("All episodes have LLM responses", empty == 0, f"{empty} empty")

    # --- Token counts present ---
    no_tokens = conn.execute(
        "SELECT COUNT(*) FROM episodes WHERE total_tokens IS NULL OR total_tokens = 0"
    ).fetchone()[0]
    check("All episodes have token counts", no_tokens == 0, f"{no_tokens} missing")

    # --- Experiment metadata ---
    meta_keys = [r[0] for r in conn.execute("SELECT key FROM experiment_metadata").fetchall()]
    check(
        "Experiment metadata stored",
        "seed" in meta_keys and "start_timestamp" in meta_keys,
        f"keys={meta_keys}",
    )

    return {"passed": all(c["passed"] for c in checks), "checks": checks}


def print_bandit_summary(conn):
    """Print bandit state for each TS condition."""
    cond_names = {2: "dimension_feedback", 3: "full_system", 4: "qualitative"}
    for cid in [2, 3, 4]:
        arms = conn.execute(
            """SELECT preset_id, alpha, beta, pulls, total_reward
               FROM bandit_state WHERE condition_id = ?
               ORDER BY (alpha / (alpha + beta)) DESC""",
            (cid,),
        ).fetchall()
        total_pulls = sum(a["pulls"] for a in arms)
        if total_pulls > 0:
            best = arms[0]
            mean = best["alpha"] / (best["alpha"] + best["beta"])
            arm_summary = ", ".join(
                f"{a['preset_id']}({a['pulls']})" for a in arms if a["pulls"] > 0
            )
            print(
                f"    Cond {cid} ({cond_names[cid]}): "
                f"best={best['preset_id']} mean={mean:.3f} | {arm_summary}"
            )
        else:
            print(f"    Cond {cid} ({cond_names[cid]}): no pulls (all parses failed)")


def print_sample_feedback(conn):
    """Print a sample of actual parsed feedback for manual inspection."""
    print("\n  Sample feedback (first parsed per condition):")
    cond_labels = {2: "likert", 3: "full+emb", 4: "qual"}
    for cid in [2, 3, 4]:
        row = conn.execute(
            """SELECT f.*, e.condition_id, e.task_id
               FROM feedback f
               JOIN episodes e ON f.episode_id = e.episode_id
               WHERE e.condition_id = ?
               LIMIT 1""",
            (cid,),
        ).fetchone()
        if row:
            label = cond_labels[cid]
            if row["rating_recency"] is not None:
                print(
                    f"    [{label}] Likert: R={row['rating_recency']} "
                    f"I={row['rating_importance']} V={row['rating_relevance']}"
                )
            if row["inferred_recency"] is not None:
                print(
                    f"    [{label}] Inferred: R={row['inferred_recency']:.3f} "
                    f"I={row['inferred_importance']:.3f} "
                    f"V={row['inferred_relevance']:.3f}"
                )
            if row["explanation"]:
                preview = row["explanation"][:150].replace("\n", " ")
                print(f"    [{label}] Explanation: {preview}...")
        else:
            print(f"    [cond {cid}] No parsed feedback")


def print_episode_summary(conn):
    """Print token usage and step counts."""
    for cid in range(1, 5):
        row = conn.execute(
            """SELECT AVG(total_tokens) as avg_tok, AVG(step_count) as avg_steps,
                      SUM(total_tokens) as sum_tok
               FROM episodes WHERE condition_id = ?""",
            (cid,),
        ).fetchone()
        if row and row["avg_tok"]:
            print(
                f"    Cond {cid}: avg_tokens={row['avg_tok']:.0f} "
                f"avg_steps={row['avg_steps']:.1f} "
                f"total_tokens={row['sum_tok']}"
            )


# ── Main ──


def main():
    parser = argparse.ArgumentParser(
        description="Integration test: end-to-end with real MiniMax API"
    )
    parser.add_argument(
        "--tasks-per-theme",
        type=int,
        default=2,
        help="Tasks per theme (default: 2, total = 5 × this)",
    )
    parser.add_argument("--db", default="integration_test.db")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument(
        "--keep-db", action="store_true", help="Keep database after test for inspection"
    )
    parser.add_argument("--corpus", default="data/tasks/corpus_v3.json")
    parser.add_argument("--library", default="data/skills/library_v3.json")
    args = parser.parse_args()

    # Preflight
    if not os.environ.get("MINIMAX_API_KEY"):
        print("Error: MINIMAX_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    n_tasks = args.tasks_per_theme * 5
    n_episodes = n_tasks * 4

    print(f"{'=' * 60}")
    print(f"Integration Test: {n_tasks} tasks × 4 conditions = {n_episodes} episodes")
    print(f"{'=' * 60}\n")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Clean stale test DB
    db_path = ROOT / args.db
    for p in [db_path, Path(f"{db_path}-wal"), Path(f"{db_path}-shm")]:
        if p.exists():
            p.unlink()

    # ── Step 1: Load data ──
    print("Step 1: Loading tasks and skills...")
    conn = init_db(str(db_path))
    tasks = select_tasks(ROOT / args.corpus, args.tasks_per_theme)
    skills = load_data(conn, tasks, ROOT / args.library)
    print(f"  {len(tasks)} tasks, {len(skills)} skills loaded\n")

    # ── Step 2: Embed skills ──
    print("Step 2: Embedding skills (first run downloads ~1.2GB model)...")
    embedder = Embedder()
    embed_skills(conn, embedder, skills)
    conn.close()
    print()

    # ── Step 3: Initialize runner ──
    print("Step 3: Initializing runner...")
    llm_client = MiniMaxClient()
    runner = ExperimentRunner(
        config_path=str(ROOT / args.config),
        db_path=str(db_path),
        llm_client=llm_client,
        embedder=embedder,
    )
    runner.initialize()
    print("  All components ready\n")

    # ── Step 4: Run experiment ──
    print(f"Step 4: Running {n_episodes} episodes...\n")
    t0 = time.time()
    runner.run()
    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.0f}s ({elapsed / 60:.1f} min)\n")

    # ── Step 5: Validate ──
    print(f"{'=' * 60}")
    print("Validation Report")
    print(f"{'=' * 60}\n")

    report = validate(runner.conn, n_tasks)

    for c in report["checks"]:
        mark = "✓" if c["passed"] else "✗"
        print(f"  {mark} {c['name']}: {c['detail']}")

    print("\n  Bandit summary:")
    print_bandit_summary(runner.conn)

    print("\n  Token usage:")
    print_episode_summary(runner.conn)

    print_sample_feedback(runner.conn)

    # ── Result ──
    print(f"\n{'=' * 60}")
    if report["passed"]:
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL")
        failed = [c["name"] for c in report["checks"] if not c["passed"]]
        for f in failed:
            print(f"  ✗ {f}")
    print(f"{'=' * 60}")

    # Cleanup
    runner.conn.close()
    if not args.keep_db and report["passed"]:
        for p in [db_path, Path(f"{db_path}-wal"), Path(f"{db_path}-shm")]:
            if p.exists():
                p.unlink()
        print("(test DB cleaned up)")
    else:
        print(f"(test DB: {db_path})")

    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
