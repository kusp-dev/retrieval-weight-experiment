#!/usr/bin/env python3
"""
Live experiment monitor — queries experiment.db and prints a dashboard.

Usage:
  python scripts/monitor.py              # one-shot summary
  python scripts/monitor.py --watch 30   # refresh every 30 seconds
  python scripts/monitor.py --full       # include per-task breakdown
"""

import argparse
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "experiment_v3.db"

COND_NAMES = {1: "control", 2: "dim_feedback", 3: "full_system", 4: "qualitative"}
TOTAL_TASKS = 300
TOTAL_EPISODES = 1200


def get_conn():
    if not DB_PATH.exists():
        print("No experiment_v3.db found. Experiment not started yet.")
        sys.exit(1)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def clear_screen():
    subprocess.run(["clear"], check=False)


def print_dashboard(conn, full=False):
    now = time.strftime("%H:%M:%S")

    # ── Overall progress ──
    total_ep = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
    last_task = conn.execute(
        "SELECT value FROM experiment_metadata WHERE key='last_completed_task'"
    ).fetchone()
    tasks_done = int(last_task["value"]) if last_task else 0
    pct = tasks_done / TOTAL_TASKS * 100

    start_row = conn.execute(
        "SELECT value FROM experiment_metadata WHERE key='start_timestamp'"
    ).fetchone()

    elapsed_str = ""
    eta_str = ""
    if start_row and tasks_done > 0:
        from datetime import datetime

        start_time = datetime.strptime(start_row["value"], "%Y-%m-%dT%H:%M:%S")
        elapsed = (datetime.now() - start_time).total_seconds()
        elapsed_str = f"{elapsed / 3600:.1f}h"
        rate = tasks_done / elapsed  # tasks per second
        remaining = (TOTAL_TASKS - tasks_done) / rate if rate > 0 else 0
        eta_str = f"{remaining / 3600:.1f}h"

    print(f"{'=' * 62}")
    print(f"  Experiment Monitor                          {now}")
    print(f"{'=' * 62}")
    print(
        f"  Tasks: {tasks_done}/{TOTAL_TASKS} ({pct:.0f}%)  "
        f"Episodes: {total_ep}/{TOTAL_EPISODES}  "
        f"Elapsed: {elapsed_str}  ETA: {eta_str}"
    )

    # Progress bar
    bar_width = 40
    filled = int(bar_width * tasks_done / TOTAL_TASKS)
    bar = "█" * filled + "░" * (bar_width - filled)
    print(f"  [{bar}]")
    print()

    # ── Per-condition stats ──
    print(
        f"  {'Condition':<16} {'Episodes':>8} {'Parse%':>7}"
        f" {'AvgReward':>10} {'AvgTokens':>10} {'AvgSteps':>9}"
    )
    print(f"  {'─' * 16} {'─' * 8} {'─' * 7} {'─' * 10} {'─' * 10} {'─' * 9}")

    for cid in range(1, 5):
        row = conn.execute(
            """SELECT COUNT(*) as n,
                      AVG(total_tokens) as avg_tok,
                      AVG(step_count) as avg_steps
               FROM episodes WHERE condition_id = ?""",
            (cid,),
        ).fetchone()

        fb_count = conn.execute(
            """SELECT COUNT(*) FROM feedback f
               JOIN episodes e ON f.episode_id = e.episode_id
               WHERE e.condition_id = ?""",
            (cid,),
        ).fetchone()[0]

        avg_reward_row = conn.execute(
            """SELECT AVG(
                   COALESCE(
                       (rating_recency + rating_importance + rating_relevance - 3) / 12.0,
                       (inferred_recency + inferred_importance + inferred_relevance) / 3.0
                   )
               ) as avg_r
               FROM feedback f
               JOIN episodes e ON f.episode_id = e.episode_id
               WHERE e.condition_id = ?""",
            (cid,),
        ).fetchone()

        n = row["n"] if row["n"] else 0
        parse_rate = fb_count / n * 100 if n > 0 else 0
        avg_tok = f"{row['avg_tok']:.0f}" if row["avg_tok"] else "-"
        avg_steps = f"{row['avg_steps']:.1f}" if row["avg_steps"] else "-"
        avg_reward = f"{avg_reward_row['avg_r']:.3f}" if avg_reward_row["avg_r"] else "-"

        if cid == 1:
            parse_rate_str = "N/A"
            avg_reward = "N/A"
        else:
            parse_rate_str = f"{parse_rate:.0f}%"

        print(
            f"  {COND_NAMES[cid]:<16} {n:>8}"
            f" {parse_rate_str:>7} {avg_reward:>10}"
            f" {avg_tok:>10} {avg_steps:>9}"
        )

    # ── Token totals ──
    total_tokens = conn.execute("SELECT SUM(total_tokens) FROM episodes").fetchone()[0]
    print(f"\n  Total tokens: {total_tokens:,}" if total_tokens else "")

    # ── Bandit state ──
    print(f"\n  {'Bandit State'}")
    print(f"  {'─' * 58}")
    for cid in [2, 3, 4]:
        arms = conn.execute(
            """SELECT preset_id, alpha, beta, pulls, total_reward
               FROM bandit_state WHERE condition_id = ?
               ORDER BY (alpha / (alpha + beta)) DESC""",
            (cid,),
        ).fetchall()
        total_pulls = sum(a["pulls"] for a in arms)
        if total_pulls > 0:
            parts = []
            for a in arms:
                mean = a["alpha"] / (a["alpha"] + a["beta"])
                marker = " <-best" if a == arms[0] else ""
                parts.append(f"{a['preset_id']}({a['pulls']}) u={mean:.2f}{marker}")
            # Show top 3 arms
            arm_str = "  ".join(parts[:3])
            print(f"  C{cid} {COND_NAMES[cid]:<14} {arm_str}")
        else:
            print(f"  C{cid} {COND_NAMES[cid]:<14} (no pulls yet)")

    # ── Recent episodes ──
    print("\n  Last 5 episodes:")
    print(f"  {'─' * 58}")
    recent = conn.execute(
        """SELECT e.condition_id, e.task_id, e.preset_id, e.step_count,
                  e.total_tokens, e.completed_at,
                  f.rating_recency, f.rating_importance, f.rating_relevance,
                  f.inferred_recency, f.inferred_importance, f.inferred_relevance
           FROM episodes e
           LEFT JOIN feedback f ON e.episode_id = f.episode_id
           ORDER BY e.episode_id DESC LIMIT 5"""
    ).fetchall()
    for r in recent:
        cname = COND_NAMES.get(r["condition_id"], "?")
        if r["rating_recency"] is not None:
            fb = f"R={r['rating_recency']} I={r['rating_importance']} V={r['rating_relevance']}"
        elif r["inferred_recency"] is not None:
            fb = (
                f"R={r['inferred_recency']:.2f}"
                f" I={r['inferred_importance']:.2f}"
                f" V={r['inferred_relevance']:.2f}"
            )
        else:
            fb = "no feedback"
        ts = r["completed_at"][-8:] if r["completed_at"] else ""
        print(f"  {ts} {cname:<14} {r['preset_id']:<20} tok={r['total_tokens']:>5} {fb}")

    # ── Condition 3 vs 2 check (key differentiator) ──
    c3_fb = conn.execute(
        """SELECT COUNT(*) FROM feedback f
           JOIN episodes e ON f.episode_id = e.episode_id
           WHERE e.condition_id = 3"""
    ).fetchone()[0]
    c3_emb = conn.execute(
        """SELECT COUNT(*) FROM feedback_embeddings fe
           JOIN feedback f ON fe.feedback_id = f.feedback_id
           JOIN episodes e ON f.episode_id = e.episode_id
           WHERE e.condition_id = 3"""
    ).fetchone()[0]
    c2_emb = conn.execute(
        """SELECT COUNT(*) FROM feedback_embeddings fe
           JOIN feedback f ON fe.feedback_id = f.feedback_id
           JOIN episodes e ON f.episode_id = e.episode_id
           WHERE e.condition_id = 2"""
    ).fetchone()[0]
    print(
        f"\n  C3 vs C2 differentiator:"
        f" C3 embeddings={c3_emb}/{c3_fb} parsed,"
        f" C2 embeddings={c2_emb} (should be 0)"
    )

    print("\n  Langfuse: https://us.cloud.langfuse.com")
    print(f"{'=' * 62}")


def main():
    parser = argparse.ArgumentParser(description="Live experiment dashboard")
    parser.add_argument(
        "--watch", type=int, metavar="SECS", help="Refresh interval in seconds (default: one-shot)"
    )
    parser.add_argument("--full", action="store_true", help="Include per-task breakdown")
    args = parser.parse_args()

    if args.watch:
        try:
            while True:
                clear_screen()
                conn = get_conn()
                print_dashboard(conn, full=args.full)
                conn.close()
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nMonitor stopped.")
    else:
        conn = get_conn()
        print_dashboard(conn, full=args.full)
        conn.close()


if __name__ == "__main__":
    main()
