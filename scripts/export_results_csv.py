#!/usr/bin/env python3
"""Export experiment results to CSV for public release.

Reads experiment_v3.db (and warm-start DB if found) and exports clean,
well-documented CSV files for reproducibility and analysis.

Output files (in data/exports/):
  - episodes.csv          — Per-episode data (condition, task, tokens, success, steps)
  - bandit_summary.csv    — Final bandit state per condition (arm probabilities, convergence)
  - per_condition_stats.csv — Aggregate statistics with bootstrap CIs and p-values

Usage:
    cd ~/retrieval-weight-experiment && uv run python scripts/export_results_csv.py
    cd ~/retrieval-weight-experiment && uv run python scripts/export_results_csv.py --include-warmstart
"""

import argparse
import csv
import sqlite3
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
V3_DB = ROOT / "experiment_v3.db"
WARMSTART_DB = ROOT / "warmstart_experiment.db"
EXPORT_DIR = ROOT / "data" / "exports"

COND_NAMES = {
    1: "C1_control",
    2: "C2_dimension_feedback",
    3: "C3_full_system",
    4: "C4_qualitative",
    6: "C6_warmstart_qual",
}


def get_db(path: Path) -> sqlite3.Connection:
    """Open a SQLite database connection."""
    if not path.exists():
        raise FileNotFoundError(f"Database not found: {path}")
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def bayesian_bootstrap_mean(data, n_boot=10000, seed=42):
    """Bayesian bootstrap for the mean. Returns (mean, ci_lower, ci_upper)."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(data, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    weights = rng.dirichlet(np.ones(n), size=n_boot)
    means = weights @ arr
    return (
        float(np.mean(means)),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def bayesian_bootstrap_p_value(data_a, data_b, n_boot=10000, seed=42):
    """Probability that B has lower mean than A (token reduction)."""
    rng = np.random.default_rng(seed)
    arr_a = np.asarray(data_a, dtype=np.float64)
    arr_b = np.asarray(data_b, dtype=np.float64)
    w_a = rng.dirichlet(np.ones(len(arr_a)), size=n_boot)
    w_b = rng.dirichlet(np.ones(len(arr_b)), size=n_boot)
    means_a = w_a @ arr_a
    means_b = w_b @ arr_b
    # Proportion of bootstrap samples where B < A
    return float(np.mean(means_b < means_a))


def export_episodes(conn, condition_ids, source_label, writer):
    """Export per-episode rows."""
    for cid in condition_ids:
        rows = conn.execute(
            """
            SELECT e.episode_id, e.condition_id, e.task_id, e.total_tokens,
                   e.success, e.step_count, e.task_order, e.preset_id,
                   e.input_tokens, e.output_tokens, e.duration_ms,
                   t.theme, t.difficulty
            FROM episodes e
            JOIN tasks t ON e.task_id = t.task_id
            WHERE e.condition_id = ?
            ORDER BY e.task_order
        """,
            (cid,),
        ).fetchall()

        for row in rows:
            writer.writerow(
                {
                    "episode_id": row["episode_id"],
                    "condition_id": row["condition_id"],
                    "condition_name": COND_NAMES.get(
                        row["condition_id"], f"C{row['condition_id']}"
                    ),
                    "source": source_label,
                    "task_id": row["task_id"],
                    "task_order": row["task_order"],
                    "theme": row["theme"],
                    "difficulty": row["difficulty"],
                    "preset_id": row["preset_id"] or "",
                    "total_tokens": row["total_tokens"],
                    "input_tokens": row["input_tokens"],
                    "output_tokens": row["output_tokens"],
                    "success": row["success"],
                    "step_count": row["step_count"],
                    "duration_ms": row["duration_ms"],
                }
            )


def export_bandit_summary(conn, condition_ids, source_label, writer):
    """Export bandit state summary per condition."""
    for cid in condition_ids:
        if cid == 1:
            # Control has no bandit
            continue

        arms = conn.execute(
            """
            SELECT preset_id, alpha, beta, pulls, total_reward
            FROM bandit_state
            WHERE condition_id = ?
            ORDER BY (alpha / (alpha + beta)) DESC
        """,
            (cid,),
        ).fetchall()

        if not arms:
            continue

        top_arm = arms[0]["preset_id"]

        for rank, arm in enumerate(arms, 1):
            mean = arm["alpha"] / (arm["alpha"] + arm["beta"])
            writer.writerow(
                {
                    "condition_id": cid,
                    "condition_name": COND_NAMES.get(cid, f"C{cid}"),
                    "source": source_label,
                    "preset_id": arm["preset_id"],
                    "rank_by_posterior": rank,
                    "alpha": round(arm["alpha"], 4),
                    "beta": round(arm["beta"], 4),
                    "posterior_mean": round(mean, 4),
                    "pulls": arm["pulls"],
                    "total_reward": round(arm["total_reward"], 4),
                    "is_best_arm": 1 if arm["preset_id"] == top_arm else 0,
                }
            )


def export_per_condition_stats(conns_with_ids, writer):
    """Export aggregate statistics per condition with bootstrap CIs and p-values.

    conns_with_ids: list of (conn, condition_ids, source_label) tuples
    """
    # First, collect all token data for the control condition (C1)
    control_tokens = None
    all_stats = []

    for conn, condition_ids, source_label in conns_with_ids:
        for cid in condition_ids:
            tokens = [
                row[0]
                for row in conn.execute(
                    "SELECT total_tokens FROM episodes WHERE condition_id = ? AND total_tokens IS NOT NULL",
                    (cid,),
                ).fetchall()
            ]

            if not tokens:
                continue

            if cid == 1 and source_label == "v3":
                control_tokens = tokens

            mean_tok, ci_low, ci_high = bayesian_bootstrap_mean(tokens)
            median_tok = float(np.median(tokens))
            std_tok = float(np.std(tokens))

            # Success rate
            success_row = conn.execute(
                "SELECT AVG(CAST(success AS REAL)) as sr FROM episodes WHERE condition_id = ?",
                (cid,),
            ).fetchone()
            success_rate = success_row["sr"] if success_row["sr"] is not None else None

            # Mean steps
            steps_row = conn.execute(
                "SELECT AVG(step_count) as ms FROM episodes WHERE condition_id = ?",
                (cid,),
            ).fetchone()
            mean_steps = steps_row["ms"]

            all_stats.append(
                {
                    "cid": cid,
                    "source": source_label,
                    "tokens": tokens,
                    "mean_tokens": round(mean_tok, 1),
                    "median_tokens": round(median_tok, 1),
                    "std_tokens": round(std_tok, 1),
                    "ci_lower": round(ci_low, 1),
                    "ci_upper": round(ci_high, 1),
                    "n_episodes": len(tokens),
                    "success_rate": round(success_rate, 4) if success_rate is not None else None,
                    "mean_steps": round(mean_steps, 2) if mean_steps is not None else None,
                }
            )

    # Now compute p-values vs control
    for stat in all_stats:
        p_vs_control = None
        reduction_vs_control = None

        if control_tokens and stat["cid"] != 1:
            p_vs_control = bayesian_bootstrap_p_value(control_tokens, stat["tokens"])
            reduction_vs_control = round(
                (np.mean(control_tokens) - stat["mean_tokens"]) / np.mean(control_tokens) * 100,
                1,
            )

        writer.writerow(
            {
                "condition_id": stat["cid"],
                "condition_name": COND_NAMES.get(stat["cid"], f"C{stat['cid']}"),
                "source": stat["source"],
                "n_episodes": stat["n_episodes"],
                "mean_tokens": stat["mean_tokens"],
                "median_tokens": stat["median_tokens"],
                "std_tokens": stat["std_tokens"],
                "ci_lower_95": stat["ci_lower"],
                "ci_upper_95": stat["ci_upper"],
                "p_value_vs_control": round(p_vs_control, 4) if p_vs_control is not None else "",
                "reduction_vs_control_pct": reduction_vs_control
                if reduction_vs_control is not None
                else "",
                "success_rate": stat["success_rate"] if stat["success_rate"] is not None else "",
                "mean_steps": stat["mean_steps"] if stat["mean_steps"] is not None else "",
            }
        )


def main():
    parser = argparse.ArgumentParser(description="Export experiment results to CSV")
    parser.add_argument(
        "--include-warmstart",
        action="store_true",
        help="Include warm-start experiment data (C6) in exports",
    )
    args = parser.parse_args()

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Exporting results to {EXPORT_DIR}/\n")

    # Open databases
    conn_v3 = get_db(V3_DB)
    v3_conditions = [1, 2, 3, 4]

    conns_with_ids = [(conn_v3, v3_conditions, "v3")]

    conn_ws = None
    ws_conditions = []
    if args.include_warmstart and WARMSTART_DB.exists():
        conn_ws = get_db(WARMSTART_DB)
        # Only include C6 (the complete warm-start condition)
        ws_conditions = [6]
        conns_with_ids.append((conn_ws, ws_conditions, "warmstart"))
        print(f"Including warm-start data: C6 ({WARMSTART_DB.name})")

    # 1. Episodes CSV
    episodes_path = EXPORT_DIR / "episodes.csv"
    fieldnames_ep = [
        "episode_id",
        "condition_id",
        "condition_name",
        "source",
        "task_id",
        "task_order",
        "theme",
        "difficulty",
        "preset_id",
        "total_tokens",
        "input_tokens",
        "output_tokens",
        "success",
        "step_count",
        "duration_ms",
    ]
    with open(episodes_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_ep)
        writer.writeheader()
        export_episodes(conn_v3, v3_conditions, "v3", writer)
        if conn_ws:
            export_episodes(conn_ws, ws_conditions, "warmstart", writer)

    n_rows = sum(1 for _ in open(episodes_path)) - 1
    print(f"  episodes.csv: {n_rows} rows")

    # 2. Bandit Summary CSV
    bandit_path = EXPORT_DIR / "bandit_summary.csv"
    fieldnames_bandit = [
        "condition_id",
        "condition_name",
        "source",
        "preset_id",
        "rank_by_posterior",
        "alpha",
        "beta",
        "posterior_mean",
        "pulls",
        "total_reward",
        "is_best_arm",
    ]
    with open(bandit_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_bandit)
        writer.writeheader()
        export_bandit_summary(conn_v3, v3_conditions, "v3", writer)
        if conn_ws:
            export_bandit_summary(conn_ws, ws_conditions, "warmstart", writer)

    n_rows = sum(1 for _ in open(bandit_path)) - 1
    print(f"  bandit_summary.csv: {n_rows} rows")

    # 3. Per-Condition Stats CSV
    stats_path = EXPORT_DIR / "per_condition_stats.csv"
    fieldnames_stats = [
        "condition_id",
        "condition_name",
        "source",
        "n_episodes",
        "mean_tokens",
        "median_tokens",
        "std_tokens",
        "ci_lower_95",
        "ci_upper_95",
        "p_value_vs_control",
        "reduction_vs_control_pct",
        "success_rate",
        "mean_steps",
    ]
    with open(stats_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_stats)
        writer.writeheader()
        export_per_condition_stats(conns_with_ids, writer)

    n_rows = sum(1 for _ in open(stats_path)) - 1
    print(f"  per_condition_stats.csv: {n_rows} rows")

    # Close connections
    conn_v3.close()
    if conn_ws:
        conn_ws.close()

    print(f"\nExport complete. Files in: {EXPORT_DIR}")


if __name__ == "__main__":
    main()
