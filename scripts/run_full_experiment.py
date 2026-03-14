#!/usr/bin/env python3
"""
Run the full experiment: 300 tasks × 4 conditions = 1,200 episodes.

Step 14 in the experiment build checklist (STATUS.md).

Supports clean resumption — if interrupted, just re-run this script.
The runner picks up from the last completed task via experiment_metadata.

Requirements:
  - .env with MINIMAX_API_KEY (required) + LANGFUSE_* keys (optional)
  - Run with .venv313/bin/python (Python 3.13 for Langfuse compatibility)
  - Qwen3-Embedding-0.6B model (auto-downloads ~1.2GB on first run)

Usage:
  .venv313/bin/python scripts/run_full_experiment.py
  .venv313/bin/python scripts/run_full_experiment.py --resume   # explicit resume
  .venv313/bin/python scripts/run_full_experiment.py --dry-run  # preflight only
"""

import argparse
import atexit
import fcntl
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Load .env before any imports that need API keys
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from src.db.schema import init_db  # noqa: E402
from src.embeddings.embedder import Embedder, store_skill_embedding  # noqa: E402
from src.experiment.runner import ExperimentRunner  # noqa: E402
from src.llm.minimax_client import MiniMaxClient  # noqa: E402

logger = logging.getLogger("experiment")

DB_PATH = ROOT / "experiment_v3.db"
CORPUS_PATH = ROOT / "data" / "tasks" / "corpus_v3.json"
SKILLS_PATH = ROOT / "data" / "skills" / "library_v3.json"
GT_PATH = ROOT / "data" / "ground_truth_v3.json"
CONFIG_PATH = ROOT / "configs" / "experiment.yaml"


def build_skill_content(skill: dict) -> str:
    """Build searchable content from skill metadata."""
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


def load_data(conn) -> tuple[int, int]:
    """Load full task corpus and skill library into DB. Returns (n_tasks, n_skills).

    v3: ground truth is in a separate file (ground_truth_v3.json), not embedded
    in the corpus. We load it separately and merge by task_id.
    """
    with open(CORPUS_PATH) as f:
        tasks = json.load(f)

    # Load ground truth from separate file and build lookup by task_id
    with open(GT_PATH) as f:
        gt_entries = json.load(f)
    gt_lookup = {entry["task_id"]: entry for entry in gt_entries}

    for task in tasks:
        gt = gt_lookup.get(task["task_id"], {})
        ground_truth_skills = gt.get("relevant_skill_ids", [])
        expected_approach = gt.get("primary_skill_id", "")
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
                json.dumps(ground_truth_skills),
                expected_approach,
            ),
        )

    with open(SKILLS_PATH) as f:
        skills = json.load(f)

    for skill in skills:
        content = build_skill_content(skill)
        conn.execute(
            """INSERT OR IGNORE INTO skills (skill_id, domain, title, content)
               VALUES (?, ?, ?, ?)""",
            (skill["skill_id"], skill["domain"], skill["title"], content),
        )

    conn.commit()

    # Rebuild FTS5 index to ensure consistency with skills table rowids
    conn.execute("INSERT INTO skills_fts(skills_fts) VALUES('rebuild')")
    conn.commit()

    return len(tasks), len(skills)


def embed_skills(conn, embedder: Embedder) -> None:
    """Embed all skills that don't already have embeddings."""
    rows = conn.execute(
        """SELECT s.skill_id, s.content FROM skills s
           LEFT JOIN skill_embeddings se ON s.skill_id = se.skill_id
           WHERE se.skill_id IS NULL"""
    ).fetchall()

    if not rows:
        logger.info("All skills already embedded, skipping")
        return

    logger.info(f"Embedding {len(rows)} skills...")
    contents = [r["content"] for r in rows]
    t0 = time.time()
    embeddings = embedder.embed_batch(contents)
    elapsed = time.time() - t0

    for row, emb in zip(rows, embeddings):
        store_skill_embedding(conn, row["skill_id"], emb)

    logger.info(f"Embedded {len(rows)} skills in {elapsed:.1f}s")


def preflight(dry_run: bool = False) -> bool:
    """Check all prerequisites before starting."""
    ok = True

    # API key
    if not os.environ.get("MINIMAX_API_KEY"):
        if dry_run:
            print("  WARN: MINIMAX_API_KEY not set (not required for dry-run)")
        else:
            print("  FAIL: MINIMAX_API_KEY not set in environment or .env")
            ok = False
    else:
        print("  OK: MINIMAX_API_KEY set")

    # Langfuse (optional)
    if os.environ.get("LANGFUSE_SECRET_KEY"):
        print("  OK: Langfuse keys set (tracing enabled)")
    else:
        print("  WARN: Langfuse keys not set (tracing disabled)")

    # Data files
    for path, label in [
        (CORPUS_PATH, "Task corpus"),
        (SKILLS_PATH, "Skill library"),
        (GT_PATH, "Ground truth"),
        (CONFIG_PATH, "Config"),
    ]:
        if path.exists():
            print(f"  OK: {label} ({path.name})")
        else:
            print(f"  FAIL: {label} not found at {path}")
            ok = False

    # Task count
    if CORPUS_PATH.exists():
        with open(CORPUS_PATH) as f:
            n = len(json.load(f))
        print(f"  OK: {n} tasks in corpus")

    return ok


LOCK_PATH = ROOT / "experiment.lock"

_lock_fd = None
_shutdown_requested = False


def acquire_lock():
    """Acquire exclusive file lock. Prevents concurrent runs / DB resets."""
    global _lock_fd
    _lock_fd = open(LOCK_PATH, "w")
    try:
        fcntl.flock(_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        # Read existing lock info
        with open(LOCK_PATH) as f:
            info = f.read().strip()
        print(f"\n  BLOCKED: Another experiment is running.\n  Lock info: {info}")
        print(f"  If stale, delete {LOCK_PATH} and retry.")
        sys.exit(1)
    # Write lock info
    _lock_fd.write(f"PID={os.getpid()} started={time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
    _lock_fd.flush()


def release_lock():
    """Release file lock and remove lockfile."""
    global _lock_fd
    if _lock_fd is not None:
        fcntl.flock(_lock_fd, fcntl.LOCK_UN)
        _lock_fd.close()
        _lock_fd = None
    if LOCK_PATH.exists():
        LOCK_PATH.unlink()


def _handle_signal(signum, frame):
    """Handle SIGTERM/SIGINT gracefully — finish current episode, then stop."""
    global _shutdown_requested
    sig_name = signal.Signals(signum).name
    logger.info(f"Received {sig_name} — finishing current task, then shutting down...")
    _shutdown_requested = True


def main():
    parser = argparse.ArgumentParser(description="Run full experiment (300 × 4 = 1,200 episodes)")
    parser.add_argument(
        "--resume", action="store_true", help="Explicit resume (auto-detected anyway)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Preflight checks only, don't run")
    parser.add_argument("--db", default=str(DB_PATH), help="Database path")
    args = parser.parse_args()

    print(f"{'=' * 60}")
    print("Retrieval Weight Experiment — Full Run")
    print(f"{'=' * 60}\n")

    # Preflight
    print("Preflight checks:")
    if not preflight(dry_run=args.dry_run):
        print("\nPreflight FAILED. Fix issues above and retry.")
        sys.exit(1)
    print()

    if args.dry_run:
        print("Dry run complete. Everything looks good.")
        sys.exit(0)

    # Acquire exclusive lock (prevents DB conflicts)
    acquire_lock()
    atexit.register(release_lock)

    # Graceful shutdown on SIGTERM/SIGINT
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(ROOT / "experiment.log"),
        ],
    )

    # Step 1: Load data
    print("Step 1: Loading tasks and skills...")
    conn = init_db(args.db)
    n_tasks, n_skills = load_data(conn)
    print(f"  {n_tasks} tasks, {n_skills} skills loaded\n")

    # Step 2: Embed skills
    print("Step 2: Embedding skills...")
    embedder = Embedder()
    embed_skills(conn, embedder)
    conn.close()
    print()

    # Step 3: Initialize runner
    print("Step 3: Initializing runner...")
    llm_client = MiniMaxClient()
    runner = ExperimentRunner(
        config_path=str(CONFIG_PATH),
        db_path=args.db,
        llm_client=llm_client,
        embedder=embedder,
    )
    runner.initialize()

    resume_point = runner.get_resume_point()
    remaining = n_tasks - resume_point
    remaining_episodes = remaining * 4

    if resume_point > 0:
        print(
            f"  Resuming from task {resume_point}/{n_tasks}"
            f" ({remaining} remaining,"
            f" {remaining_episodes} episodes)"
        )
    else:
        print(f"  Fresh start: {n_tasks} tasks × 4 conditions = {n_tasks * 4} episodes")
    print()

    # Step 4: Run
    print("Step 4: Running experiment...\n")
    t0 = time.time()

    try:
        runner.run(should_stop=lambda: _shutdown_requested)
    except KeyboardInterrupt:
        pass

    if _shutdown_requested or runner.get_resume_point() < n_tasks:
        elapsed = time.time() - t0
        completed = runner.get_resume_point() - resume_point
        logger.info(
            f"Stopped after {completed} tasks ({elapsed / 60:.1f} min). "
            f"Re-run to resume from task {runner.get_resume_point()}."
        )
        runner.tracer.shutdown()
        release_lock()
        sys.exit(0)

    elapsed = time.time() - t0

    # Step 5: Summary
    print(f"\n{'=' * 60}")
    print(f"Experiment complete in {elapsed / 3600:.1f} hours ({elapsed / 60:.1f} min)")
    print(f"{'=' * 60}\n")

    # Quick stats
    total_episodes = runner.conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
    total_feedback = runner.conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
    total_tokens = runner.conn.execute("SELECT SUM(total_tokens) FROM episodes").fetchone()[0]

    print(f"  Episodes: {total_episodes}")
    print(f"  Feedback parsed: {total_feedback}")
    print(f"  Total tokens: {total_tokens:,}")

    for cid in range(1, 5):
        row = runner.conn.execute(
            """SELECT COUNT(*) as n,
                      AVG(total_tokens) as avg_tok,
                      AVG(step_count) as avg_steps
               FROM episodes WHERE condition_id = ?""",
            (cid,),
        ).fetchone()
        fb = runner.conn.execute(
            """SELECT COUNT(*) FROM feedback f
               JOIN episodes e ON f.episode_id = e.episode_id
               WHERE e.condition_id = ?""",
            (cid,),
        ).fetchone()[0]
        parse_rate = fb / row["n"] if row["n"] > 0 else 0
        print(
            f"  Cond {cid}: {row['n']} episodes, "
            f"avg_tokens={row['avg_tok']:.0f}, "
            f"avg_steps={row['avg_steps']:.1f}, "
            f"parse_rate={parse_rate:.0%}"
        )

    # Bandit summary
    print("\n  Bandit state:")
    for cid in [2, 3, 4]:
        arms = runner.conn.execute(
            """SELECT preset_id, alpha, beta, pulls
               FROM bandit_state WHERE condition_id = ?
               ORDER BY (alpha / (alpha + beta)) DESC""",
            (cid,),
        ).fetchall()
        total_pulls = sum(a["pulls"] for a in arms)
        if total_pulls > 0:
            best = arms[0]
            mean = best["alpha"] / (best["alpha"] + best["beta"])
            print(f"    Cond {cid}: best={best['preset_id']} mean={mean:.3f} pulls={total_pulls}")

    runner.tracer.shutdown()
    print(f"\nDatabase: {args.db}")
    print(f"Log: {ROOT / 'experiment.log'}")


if __name__ == "__main__":
    main()
