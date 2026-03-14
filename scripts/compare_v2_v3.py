#!/usr/bin/env python3
"""
Compare v2 vs v3 experiment results — the key question:
Does domain-specific skill desaturation make weight learning differentiate retrieval?

v2 finding: Jaccard 0.84-0.91, recency/importance saturated, weight presets equivalent.
v3 hypothesis: With 205 skills (vs 50), desaturated dimensions → presets SHOULD produce
different top-5 sets → Jaccard should drop below 0.60.

Usage:
  .venv313/bin/python scripts/compare_v2_v3.py
  .venv313/bin/python scripts/compare_v2_v3.py --v2 experiment.db --v3 experiment_v3.db
"""

import argparse
import sqlite3
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

COND_NAMES = {1: "C1 Control", 2: "C2 Dimension", 3: "C3 Full System", 4: "C4 Qualitative"}


def get_db(path):
    if not Path(path).exists():
        print(f"[ERROR] DB not found: {path}")
        return None
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def version_summary(conn, label):
    """Print summary statistics for one experiment version."""
    n_tasks = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
    n_skills = conn.execute("SELECT COUNT(*) FROM skills").fetchone()[0]
    n_episodes = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
    n_presets = conn.execute("SELECT COUNT(*) FROM weight_presets").fetchone()[0]

    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")
    print(
        f"  Tasks: {n_tasks}  |  Skills: {n_skills}"
        f"  |  Presets: {n_presets}  |  Episodes: {n_episodes}"
    )

    # Per-condition summary
    print(
        f"\n  {'Condition':<20} {'N':>5} {'Avg Tokens':>12}"
        f" {'Avg Steps':>10} {'Multi%':>8} {'Median Tok':>12}"
    )
    print(f"  {'-' * 67}")

    for cid in range(1, 5):
        row = conn.execute(
            """
            SELECT COUNT(*) as n, AVG(total_tokens) as avg_tok,
                   AVG(step_count) as avg_steps,
                   SUM(CASE WHEN step_count > 1 THEN 1 ELSE 0 END) as multi
            FROM episodes WHERE condition_id = ?
        """,
            (cid,),
        ).fetchone()

        if row["n"] == 0:
            continue

        # Median tokens
        tokens = [
            r[0]
            for r in conn.execute(
                "SELECT total_tokens FROM episodes WHERE condition_id = ? ORDER BY total_tokens",
                (cid,),
            ).fetchall()
        ]
        median_tok = np.median(tokens) if tokens else 0

        multi_pct = row["multi"] / row["n"] * 100
        print(
            f"  {COND_NAMES[cid]:<20} {row['n']:>5} {row['avg_tok']:>12,.0f} "
            f"{row['avg_steps']:>10.2f} {multi_pct:>7.1f}% {median_tok:>12,.0f}"
        )


def dimension_spread(conn, label):
    """Analyze dimension score variance — the saturation diagnostic."""
    print(f"\n  Dimension Score Spread ({label}):")
    for dim in ["recency_score", "importance_score", "relevance_score"]:
        row = conn.execute(f"""
            SELECT AVG({dim}) as mean,
                   MIN({dim}) as min, MAX({dim}) as max,
                   AVG({dim} * {dim}) - AVG({dim}) * AVG({dim}) as var
            FROM retrieval_results
        """).fetchone()
        if row["mean"] is not None:
            std = row["var"] ** 0.5 if row["var"] and row["var"] > 0 else 0
            spread = row["max"] - row["min"]
            dim_name = dim.replace("_score", "")
            print(
                f"    {dim_name:>12}: mean={row['mean']:.3f}  std={std:.3f}  "
                f"range=[{row['min']:.3f}, {row['max']:.3f}]  spread={spread:.3f}"
            )


def jaccard_analysis(conn, label):
    """Compute Jaccard similarity between condition pairs."""
    print(f"\n  Jaccard Similarity ({label}):")

    task_ids = [
        r[0]
        for r in conn.execute("SELECT DISTINCT task_id FROM episodes ORDER BY task_id").fetchall()
    ]

    for c1, c2 in combinations(range(1, 5), 2):
        jaccards = []
        for tid in task_ids:
            skills_1 = set(
                r[0]
                for r in conn.execute(
                    """SELECT skill_id FROM retrieval_results
                   WHERE episode_id = (SELECT episode_id FROM episodes
                                       WHERE condition_id = ? AND task_id = ?)
                   ORDER BY rank LIMIT 5""",
                    (c1, tid),
                ).fetchall()
            )
            skills_2 = set(
                r[0]
                for r in conn.execute(
                    """SELECT skill_id FROM retrieval_results
                   WHERE episode_id = (SELECT episode_id FROM episodes
                                       WHERE condition_id = ? AND task_id = ?)
                   ORDER BY rank LIMIT 5""",
                    (c2, tid),
                ).fetchall()
            )
            if skills_1 and skills_2:
                jaccard = len(skills_1 & skills_2) / len(skills_1 | skills_2)
                jaccards.append(jaccard)

        if jaccards:
            mean_j = np.mean(jaccards)
            perfect = sum(1 for j in jaccards if j == 1.0) / len(jaccards) * 100
            print(
                f"    C{c1} vs C{c2}: Jaccard={mean_j:.3f}"
                f"  perfect={perfect:.0f}%"
                f"  n={len(jaccards)}"
            )


def bandit_comparison(conn, label):
    """Compare bandit convergence across conditions."""
    print(f"\n  Bandit State ({label}):")
    for cid in [2, 3, 4]:
        arms = conn.execute(
            """
            SELECT preset_id, alpha, beta, pulls,
                   alpha / (alpha + beta) as mean
            FROM bandit_state WHERE condition_id = ?
            ORDER BY mean DESC
        """,
            (cid,),
        ).fetchall()

        if not arms or all(a["pulls"] == 0 for a in arms):
            print(f"    {COND_NAMES[cid]}: no data")
            continue

        total_pulls = sum(a["pulls"] for a in arms)
        best = arms[0]
        worst = arms[-1]
        spread = best["mean"] - worst["mean"]
        print(
            f"    {COND_NAMES[cid]}: best={best['preset_id']}({best['mean']:.3f}) "
            f"spread={spread:.3f} pulls={total_pulls}"
        )


def bootstrap_comparison(conn_v2, conn_v3):
    """Bootstrap test: is v3's treatment effect different from v2's?"""
    print(f"\n{'═' * 60}")
    print("  CROSS-VERSION COMPARISON")
    print(f"{'═' * 60}")

    for cid in range(1, 5):
        v2_tokens = [
            r[0]
            for r in conn_v2.execute(
                "SELECT total_tokens FROM episodes WHERE condition_id = ?",
                (cid,),
            ).fetchall()
        ]
        v3_tokens = [
            r[0]
            for r in conn_v3.execute(
                "SELECT total_tokens FROM episodes WHERE condition_id = ?",
                (cid,),
            ).fetchall()
        ]

        if not v2_tokens or not v3_tokens:
            continue

        v2_mean = np.mean(v2_tokens)
        v3_mean = np.mean(v3_tokens)
        diff = v2_mean - v3_mean
        pct = diff / v2_mean * 100 if v2_mean > 0 else 0

        print(f"\n  {COND_NAMES[cid]}:")
        print(
            f"    v2 mean: {v2_mean:,.0f}  |  v3 mean: {v3_mean:,.0f}  |  "
            f"diff: {diff:+,.0f} ({pct:+.1f}%)"
        )

    # The KEY question: does v3 show differential Jaccard?
    print("\n  KEY HYPOTHESIS: Desaturation → Lower Jaccard")

    for label, conn in [("v2", conn_v2), ("v3", conn_v3)]:
        task_ids = [
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT task_id FROM episodes ORDER BY task_id"
            ).fetchall()
        ]

        # C1 vs C4 Jaccard (most different conditions)
        jaccards = []
        for tid in task_ids:
            s1 = set(
                r[0]
                for r in conn.execute(
                    """SELECT skill_id FROM retrieval_results
                   WHERE episode_id = (SELECT episode_id FROM episodes
                                       WHERE condition_id = 1 AND task_id = ?)
                   ORDER BY rank LIMIT 5""",
                    (tid,),
                ).fetchall()
            )
            s4 = set(
                r[0]
                for r in conn.execute(
                    """SELECT skill_id FROM retrieval_results
                   WHERE episode_id = (SELECT episode_id FROM episodes
                                       WHERE condition_id = 4 AND task_id = ?)
                   ORDER BY rank LIMIT 5""",
                    (tid,),
                ).fetchall()
            )
            if s1 and s4:
                jaccards.append(len(s1 & s4) / len(s1 | s4))

        if jaccards:
            print(f"    {label} C1 vs C4 Jaccard: {np.mean(jaccards):.3f} (n={len(jaccards)})")


def per_difficulty_comparison(conn_v2, conn_v3):
    """Compare treatment effects by difficulty across versions."""
    print("\n  Treatment Effect by Difficulty (C1 vs C4 token reduction %):")
    print(f"    {'Difficulty':<10} {'v2':>8} {'v3':>8} {'Change':>10}")
    print(f"    {'-' * 38}")

    for diff in ["easy", "medium", "hard"]:
        v2_c1 = conn_v2.execute(
            """SELECT AVG(total_tokens) FROM episodes e
               JOIN tasks t ON e.task_id = t.task_id
               WHERE e.condition_id = 1 AND t.difficulty = ?""",
            (diff,),
        ).fetchone()[0]
        v2_c4 = conn_v2.execute(
            """SELECT AVG(total_tokens) FROM episodes e
               JOIN tasks t ON e.task_id = t.task_id
               WHERE e.condition_id = 4 AND t.difficulty = ?""",
            (diff,),
        ).fetchone()[0]
        v3_c1 = conn_v3.execute(
            """SELECT AVG(total_tokens) FROM episodes e
               JOIN tasks t ON e.task_id = t.task_id
               WHERE e.condition_id = 1 AND t.difficulty = ?""",
            (diff,),
        ).fetchone()[0]
        v3_c4 = conn_v3.execute(
            """SELECT AVG(total_tokens) FROM episodes e
               JOIN tasks t ON e.task_id = t.task_id
               WHERE e.condition_id = 4 AND t.difficulty = ?""",
            (diff,),
        ).fetchone()[0]

        v2_red = (1 - v2_c4 / v2_c1) * 100 if v2_c1 and v2_c4 else 0
        v3_red = (1 - v3_c4 / v3_c1) * 100 if v3_c1 and v3_c4 else 0
        change = v3_red - v2_red

        print(f"    {diff:<10} {v2_red:>7.1f}% {v3_red:>7.1f}% {change:>+9.1f}pp")


def main():
    parser = argparse.ArgumentParser(description="Compare v2 vs v3 experiment results")
    parser.add_argument("--v2", default=str(ROOT / "experiment.db"), help="v2 database")
    parser.add_argument("--v3", default=str(ROOT / "experiment_v3.db"), help="v3 database")
    args = parser.parse_args()

    print(f"{'═' * 60}")
    print("  v2 vs v3 EXPERIMENT COMPARISON")
    print(f"{'═' * 60}")

    conn_v2 = get_db(args.v2)
    conn_v3 = get_db(args.v3)

    if not conn_v2 or not conn_v3:
        sys.exit(1)

    # Check if v3 has enough data
    v3_episodes = conn_v3.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
    if v3_episodes == 0:
        print("\n  v3 has no episodes yet. Run the experiment first.")
        print("  Running v2-only analysis...\n")

    # Per-version summaries
    for conn, label in [
        (conn_v2, "v2 (50 skills, 250 tasks, 5 presets)"),
        (conn_v3, "v3 (205 skills, 300 tasks, 12 presets)"),
    ]:
        episodes = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        if episodes > 0:
            version_summary(conn, label)
            dimension_spread(conn, label)
            jaccard_analysis(conn, label)
            bandit_comparison(conn, label)

    # Cross-version comparison (only if v3 has data)
    if v3_episodes > 0:
        bootstrap_comparison(conn_v2, conn_v3)
        per_difficulty_comparison(conn_v2, conn_v3)

    conn_v2.close()
    conn_v3.close()

    print(f"\n{'═' * 60}")
    print("  DONE")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
