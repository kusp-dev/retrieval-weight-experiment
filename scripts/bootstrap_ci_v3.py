"""
BCa bootstrap confidence intervals for all condition pair differences.

Computes 95% and 99% BCa CIs for mean differences in:
  - total_tokens, output_tokens, step_count, duration_ms, success rate

Database: experiment_v3.db (v3 re-run, 300 episodes per condition)
"""

import sqlite3
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import bootstrap

DB_PATH = Path(__file__).resolve().parent.parent / "experiment_v3.db"
METRICS = ["total_tokens", "output_tokens", "step_count", "duration_ms", "success"]
N_RESAMPLES = 10_000
RNG_SEED = 42


def load_data(db_path: str) -> dict[int, dict[str, np.ndarray]]:
    """Load per-condition arrays for each metric."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    data: dict[int, dict[str, np.ndarray]] = {}
    for cid in range(1, 5):
        cols = ", ".join(METRICS)
        cur.execute(
            f"SELECT {cols} FROM episodes WHERE condition_id = ? ORDER BY episode_id",
            (cid,),
        )
        rows = cur.fetchall()
        data[cid] = {}
        for i, metric in enumerate(METRICS):
            data[cid][metric] = np.array([r[i] for r in rows], dtype=float)
    conn.close()
    return data


def mean_diff(x: np.ndarray, y: np.ndarray, axis: int = -1) -> np.ndarray:
    """Statistic function for bootstrap: mean(x) - mean(y)."""
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)


def compute_ci(
    arr_a: np.ndarray,
    arr_b: np.ndarray,
    confidence: float,
    n_resamples: int = N_RESAMPLES,
    seed: int = RNG_SEED,
) -> tuple[float, float, float]:
    """Return (point_estimate, ci_low, ci_high) for mean(a) - mean(b)."""
    point = float(np.mean(arr_a) - np.mean(arr_b))
    rng = np.random.default_rng(seed)
    try:
        res = bootstrap(
            (arr_a, arr_b),
            statistic=mean_diff,
            n_resamples=n_resamples,
            confidence_level=confidence,
            method="BCa",
            random_state=rng,
            paired=False,
        )
        ci_low = float(res.confidence_interval.low)
        ci_high = float(res.confidence_interval.high)
    except Exception as e:
        print(f"  WARNING: bootstrap failed ({e}), falling back to percentile", file=sys.stderr)
        ci_low, ci_high = float("nan"), float("nan")
    return point, ci_low, ci_high


def main() -> None:
    data = load_data(str(DB_PATH))
    pairs = list(combinations(range(1, 5), 2))

    # Print sample sizes
    print("=" * 90)
    print("BCa BOOTSTRAP CONFIDENCE INTERVALS — v3 experiment (10,000 resamples)")
    print("=" * 90)
    for cid in range(1, 5):
        n = len(data[cid]["total_tokens"])
        mean_tok = np.mean(data[cid]["total_tokens"])
        print(f"  C{cid}: n={n}, mean_total_tokens={mean_tok:.1f}")
    print()

    # For each metric
    for metric in METRICS:
        print("=" * 90)
        metric_label = metric.replace("_", " ").title()
        if metric == "success":
            metric_label = "Success Rate"
        print(f"METRIC: {metric_label}")
        print("=" * 90)
        print(
            f"{'Pair':<10} {'Point Est':>12} "
            f"{'95% CI Low':>12} {'95% CI High':>12} {'95% Sig?':>10} "
            f"{'99% CI Low':>12} {'99% CI High':>12} {'99% Sig?':>10}"
        )
        print("-" * 90)

        for ca, cb in pairs:
            arr_a = data[ca][metric]
            arr_b = data[cb][metric]

            # Check if there's any variation at all (for success rate)
            if metric == "success" and np.std(arr_a) == 0 and np.std(arr_b) == 0:
                diff = float(np.mean(arr_a) - np.mean(arr_b))
                print(
                    f"C{ca}-C{cb}    {diff:>12.4f} "
                    f"{'N/A (no var)':>25} {'---':>10} "
                    f"{'N/A (no var)':>25} {'---':>10}"
                )
                continue

            pt95, lo95, hi95 = compute_ci(arr_a, arr_b, 0.95)
            pt99, lo99, hi99 = compute_ci(arr_a, arr_b, 0.99)

            sig95 = "YES" if (lo95 > 0 or hi95 < 0) else "no"
            sig99 = "YES" if (lo99 > 0 or hi99 < 0) else "no"

            # Format based on metric
            if metric == "success":
                fmt = ".4f"
            elif metric == "duration_ms":
                fmt = ".1f"
            else:
                fmt = ".1f"

            print(
                f"C{ca}-C{cb}    {pt95:>12{fmt}} "
                f"{lo95:>12{fmt}} {hi95:>12{fmt}} {sig95:>10} "
                f"{lo99:>12{fmt}} {hi99:>12{fmt}} {sig99:>10}"
            )

        print()

    # Summary: which pairs are significant at 95% for each metric
    print("=" * 90)
    print("SUMMARY: Significant differences (CI excludes zero)")
    print("=" * 90)
    print(f"{'Metric':<20} {'95% Significant Pairs':<40} {'99% Significant Pairs':<40}")
    print("-" * 90)

    for metric in METRICS:
        metric_label = metric.replace("_", " ").title()
        if metric == "success":
            metric_label = "Success Rate"

        sig95_pairs = []
        sig99_pairs = []

        for ca, cb in pairs:
            arr_a = data[ca][metric]
            arr_b = data[cb][metric]

            if metric == "success" and np.std(arr_a) == 0 and np.std(arr_b) == 0:
                continue

            pt, lo95, hi95 = compute_ci(arr_a, arr_b, 0.95)
            _, lo99, hi99 = compute_ci(arr_a, arr_b, 0.99)

            direction = "+" if pt > 0 else "-"
            label = f"C{ca}-C{cb}({direction})"

            if lo95 > 0 or hi95 < 0:
                sig95_pairs.append(label)
            if lo99 > 0 or hi99 < 0:
                sig99_pairs.append(label)

        s95 = ", ".join(sig95_pairs) if sig95_pairs else "none"
        s99 = ", ".join(sig99_pairs) if sig99_pairs else "none"
        print(f"{metric_label:<20} {s95:<40} {s99:<40}")

    print()


if __name__ == "__main__":
    main()
