#!/usr/bin/env python3
"""
Phase 4 — Comprehensive post-experiment analysis (v3).

Run after the 300-task experiment completes to produce:
  - Primary metrics table (stdout + CSV + LaTeX)
  - Statistical significance tests (bootstrap, Mann-Whitney U, Cohen's d)
  - Convergence analysis (arm stabilization, exploration ratio over time)
  - Theme breakdown (per-theme x per-condition reward matrix)
  - Per-difficulty analysis (50 Easy / 100 Medium / 150 Hard)
  - Per-domain analysis (10 skill domains)
  - Feedback modality comparison (C2 vs C3, C2/C3 vs C4)
  - Ground truth hit rate with graded relevance (using ground_truth_v3.json)
  - NDCG@5 and MRR against ground truth
  - Jaccard similarity tracking (do conditions retrieve different skills?)
  - Token input/output breakdown (where do token savings come from?)
  - C4 anchor spread validation (are anchor-inferred scores well-distributed?)
  - Bandit convergence visualization (posterior distributions over 12 arms)
  - Publication-quality figures (300 DPI, LaTeX-ready)
  - LaTeX-ready tables

Usage:
  .venv313/bin/python scripts/analyze_experiment.py
  .venv313/bin/python scripts/analyze_experiment.py --db experiment_run2.db
"""

import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Optional: scipy for statistical tests
try:
    from scipy import stats as sp_stats

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[WARN] scipy not installed — statistical tests will be skipped.")

ROOT = Path(__file__).resolve().parent.parent

# ── Constants matching dashboard.py ──────────────────────────────────────────

COND_NAMES = {1: "C1 Control", 2: "C2 Dim Feedback", 3: "C3 Full System", 4: "C4 Qualitative"}
COND_SHORT = {1: "C1", 2: "C2", 3: "C3", 4: "C4"}
COND_COLORS = {1: "#9ca3af", 2: "#60a5fa", 3: "#34d399", 4: "#a78bfa"}
COND_COLORS_RGB = {
    1: (0.612, 0.639, 0.686),
    2: (0.376, 0.647, 0.980),
    3: (0.204, 0.827, 0.600),
    4: (0.655, 0.545, 0.980),
}

# Dark theme matching the dashboard
DARK_BG = "#0a0a0a"
DARK_FG = "#ededed"
DARK_GRID = "#1f1f1f"
DARK_ACCENT = "#2a2a2a"


def setup_dark_theme():
    """Configure matplotlib with the dashboard dark theme."""
    plt.rcParams.update(
        {
            "figure.facecolor": DARK_BG,
            "axes.facecolor": DARK_BG,
            "axes.edgecolor": DARK_GRID,
            "axes.labelcolor": DARK_FG,
            "axes.grid": True,
            "grid.color": DARK_GRID,
            "grid.alpha": 0.5,
            "text.color": DARK_FG,
            "xtick.color": DARK_FG,
            "ytick.color": DARK_FG,
            "legend.facecolor": DARK_ACCENT,
            "legend.edgecolor": DARK_GRID,
            "legend.labelcolor": DARK_FG,
            "savefig.facecolor": DARK_BG,
            "savefig.edgecolor": DARK_BG,
            "savefig.dpi": 300,
            "figure.figsize": (12, 7),
            "font.size": 11,
            "font.family": "serif",
        }
    )


# ── Database helpers ─────────────────────────────────────────────────────────


def get_db(db_path: str) -> sqlite3.Connection:
    path = Path(db_path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        print(f"[ERROR] Database not found: {path}")
        sys.exit(1)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def get_themes(conn):
    return [
        r["theme"]
        for r in conn.execute("SELECT DISTINCT theme FROM tasks ORDER BY theme").fetchall()
    ]


# ── Reward extraction ────────────────────────────────────────────────────────


def _reward_sql():
    """Standard composite reward expression (matches dashboard.py)."""
    return """COALESCE(
        (f.rating_recency + f.rating_importance + f.rating_relevance - 3) / 12.0,
        (f.inferred_recency + f.inferred_importance + f.inferred_relevance) / 3.0
    )"""


def get_rewards_per_condition(conn) -> dict[int, list[float]]:
    """Return {condition_id: [reward, ...]} for all episodes with feedback."""
    rewards = {cid: [] for cid in range(1, 5)}
    rows = conn.execute(f"""
        SELECT e.condition_id, {_reward_sql()} as reward
        FROM feedback f
        JOIN episodes e ON f.episode_id = e.episode_id
        ORDER BY e.condition_id, e.task_order
    """).fetchall()
    for r in rows:
        if r["reward"] is not None:
            rewards[r["condition_id"]].append(r["reward"])
    return rewards


def get_rewards_per_condition_ordered(conn) -> dict[int, list[dict]]:
    """Return {condition_id: [{task_order, reward}, ...]} ordered."""
    rewards = {cid: [] for cid in range(1, 5)}
    rows = conn.execute(f"""
        SELECT e.condition_id, e.task_order, {_reward_sql()} as reward
        FROM feedback f
        JOIN episodes e ON f.episode_id = e.episode_id
        ORDER BY e.condition_id, e.task_order
    """).fetchall()
    for r in rows:
        if r["reward"] is not None:
            rewards[r["condition_id"]].append(
                {
                    "task_order": r["task_order"],
                    "reward": r["reward"],
                }
            )
    return rewards


# ── 1. Primary Metrics ──────────────────────────────────────────────────────


def compute_primary_metrics(conn) -> list[dict]:
    """Compute all primary metrics per condition."""
    metrics = []
    rewards_map = get_rewards_per_condition(conn)

    for cid in range(1, 5):
        row = conn.execute(
            "SELECT COUNT(*) as n, AVG(total_tokens) as avg_tok, "
            "AVG(step_count) as avg_steps, SUM(total_tokens) as sum_tok "
            "FROM episodes WHERE condition_id = ?",
            (cid,),
        ).fetchone()

        fb_count = conn.execute(
            "SELECT COUNT(*) FROM feedback f JOIN episodes e ON f.episode_id = e.episode_id "
            "WHERE e.condition_id = ?",
            (cid,),
        ).fetchone()[0]

        n = row["n"] or 0
        rews = rewards_map.get(cid, [])
        mean_reward = float(np.mean(rews)) if rews else None

        # NDCG@5, MRR
        ir = compute_ndcg_mrr(conn, cid)

        # Bandit state (C2-C4 only)
        best_arm = None
        best_mean = None
        if cid > 1:
            arms = conn.execute(
                "SELECT preset_id, alpha, beta, pulls FROM bandit_state "
                "WHERE condition_id = ? ORDER BY (alpha / (alpha + beta)) DESC",
                (cid,),
            ).fetchall()
            if arms:
                a = arms[0]
                best_arm = a["preset_id"]
                best_mean = round(a["alpha"] / (a["alpha"] + a["beta"]), 4)

        metrics.append(
            {
                "condition": COND_NAMES[cid],
                "cid": cid,
                "episodes": n,
                "mean_reward": round(mean_reward, 4) if mean_reward is not None else None,
                "mean_steps": round(row["avg_steps"], 2) if row["avg_steps"] else None,
                "mean_tokens": round(row["avg_tok"]) if row["avg_tok"] else None,
                "total_tokens": row["sum_tok"] or 0,
                "ndcg5": ir["ndcg5"],
                "mrr": ir["mrr"],
                "parse_rate": round(fb_count / n * 100, 1) if n > 0 else 0.0,
                "best_arm": best_arm,
                "best_arm_posterior": best_mean,
            }
        )
    return metrics


def compute_ndcg_mrr(conn, condition_id=None, task_ids=None) -> dict:
    """NDCG@5 and MRR (reused from dashboard.py patterns)."""
    query = """
        SELECT e.episode_id, e.task_id, t.ground_truth_skills
        FROM episodes e
        JOIN tasks t ON e.task_id = t.task_id
        WHERE t.ground_truth_skills IS NOT NULL
    """
    params = []
    if condition_id:
        query += " AND e.condition_id = ?"
        params.append(condition_id)
    if task_ids:
        ph = ",".join("?" * len(task_ids))
        query += f" AND e.task_id IN ({ph})"
        params.extend(task_ids)

    episodes = conn.execute(query, params).fetchall()
    if not episodes:
        return {"ndcg5": None, "mrr": None, "count": 0}

    ndcg_scores, rr_scores = [], []

    for ep in episodes:
        try:
            gt_skills = json.loads(ep["ground_truth_skills"])
        except (json.JSONDecodeError, TypeError):
            continue
        if not gt_skills:
            continue

        gt_set = set(gt_skills)
        retrieved = conn.execute(
            "SELECT skill_id FROM retrieval_results WHERE episode_id = ? ORDER BY rank",
            (ep["episode_id"],),
        ).fetchall()
        if not retrieved:
            continue

        # NDCG@5
        k = min(5, len(retrieved))
        dcg, idcg = 0.0, 0.0
        for i in range(k):
            rel = 1.0 if retrieved[i]["skill_id"] in gt_set else 0.0
            dcg += rel / math.log2(i + 2)
        for i in range(min(k, len(gt_skills))):
            idcg += 1.0 / math.log2(i + 2)
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

        # MRR
        rr = 0.0
        for i, r in enumerate(retrieved):
            if r["skill_id"] in gt_set:
                rr = 1.0 / (i + 1)
                break
        rr_scores.append(rr)

    return {
        "ndcg5": round(float(np.mean(ndcg_scores)), 4) if ndcg_scores else None,
        "mrr": round(float(np.mean(rr_scores)), 4) if rr_scores else None,
        "count": len(ndcg_scores),
    }


# ── 2. Statistical Tests ────────────────────────────────────────────────────


def bayesian_bootstrap_mean(data, n_boot=10000, rng=None):
    """Bayesian bootstrap for the mean: draw Dirichlet weights and compute
    weighted mean. Returns (mean, 2.5th, 97.5th percentiles)."""
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(data)
    arr = np.asarray(data, dtype=np.float64)
    # Dirichlet(1, 1, ..., 1) weights
    weights = rng.dirichlet(np.ones(n), size=n_boot)  # (n_boot, n)
    means = weights @ arr  # (n_boot,)
    return (
        float(np.mean(means)),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def bayesian_bootstrap_diff(data_a, data_b, n_boot=10000, rng=None):
    """Bayesian bootstrap for the difference in means (B - A).
    Returns (mean_diff, ci_low, ci_high, prob_b_better)."""
    if rng is None:
        rng = np.random.default_rng(42)
    arr_a = np.asarray(data_a, dtype=np.float64)
    arr_b = np.asarray(data_b, dtype=np.float64)
    w_a = rng.dirichlet(np.ones(len(arr_a)), size=n_boot)
    w_b = rng.dirichlet(np.ones(len(arr_b)), size=n_boot)
    means_a = w_a @ arr_a
    means_b = w_b @ arr_b
    diffs = means_b - means_a
    prob_b_better = float(np.mean(diffs > 0))
    return (
        float(np.mean(diffs)),
        float(np.percentile(diffs, 2.5)),
        float(np.percentile(diffs, 97.5)),
        prob_b_better,
    )


def cohens_d(a, b):
    """Cohen's d effect size."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None
    mean_a, mean_b = np.mean(a), np.mean(b)
    var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_std = np.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2))
    if pooled_std == 0:
        return 0.0
    return float((mean_b - mean_a) / pooled_std)


def run_statistical_tests(conn) -> dict:
    """Run all statistical comparisons: each of C2, C3, C4 vs C1."""
    if not HAS_SCIPY:
        return {"error": "scipy not installed"}

    rewards = get_rewards_per_condition(conn)
    control = rewards[1]
    if len(control) < 2:
        return {"error": "Not enough control data for statistical tests"}

    rng = np.random.default_rng(42)
    results = {}

    for cid in [2, 3, 4]:
        treatment = rewards[cid]
        if len(treatment) < 2:
            results[COND_SHORT[cid]] = {"error": "Not enough data"}
            continue

        # Bayesian bootstrap difference
        diff_mean, diff_lo, diff_hi, prob_better = bayesian_bootstrap_diff(
            control, treatment, n_boot=10000, rng=rng
        )

        # Mann-Whitney U
        u_stat, u_p = sp_stats.mannwhitneyu(control, treatment, alternative="two-sided")

        # Cohen's d
        d = cohens_d(control, treatment)

        # Bootstrap 95% CI for each condition's mean
        _, c1_lo, c1_hi = bayesian_bootstrap_mean(control, n_boot=10000, rng=rng)
        _, ct_lo, ct_hi = bayesian_bootstrap_mean(treatment, n_boot=10000, rng=rng)

        results[COND_SHORT[cid]] = {
            "n_control": len(control),
            "n_treatment": len(treatment),
            "mean_control": round(float(np.mean(control)), 4),
            "mean_treatment": round(float(np.mean(treatment)), 4),
            "diff_mean": round(diff_mean, 4),
            "diff_ci_95": [round(diff_lo, 4), round(diff_hi, 4)],
            "prob_treatment_better": round(prob_better, 4),
            "mann_whitney_U": round(float(u_stat), 2),
            "mann_whitney_p": round(float(u_p), 6),
            "cohens_d": round(d, 4) if d is not None else None,
            "control_ci_95": [round(c1_lo, 4), round(c1_hi, 4)],
            "treatment_ci_95": [round(ct_lo, 4), round(ct_hi, 4)],
        }

    # Multiple comparison correction (Bonferroni on Mann-Whitney p-values)
    p_values = []
    keys = []
    for k, v in results.items():
        if isinstance(v, dict) and "mann_whitney_p" in v:
            p_values.append(v["mann_whitney_p"])
            keys.append(k)

    if p_values:
        n_tests = len(p_values)
        for i, k in enumerate(keys):
            p_adj = min(p_values[i] * n_tests, 1.0)
            results[k]["mann_whitney_p_bonferroni"] = round(p_adj, 6)
            results[k]["significant_bonferroni_05"] = p_adj < 0.05

        # Also Benjamini-Hochberg (step-up procedure)
        sorted_idx = np.argsort(p_values)
        bh_adj = np.zeros(n_tests)
        for rank_i, orig_i in enumerate(sorted_idx):
            bh_adj[orig_i] = p_values[orig_i] * n_tests / (rank_i + 1)
        # Enforce monotonicity: walk from largest rank down, carry minimum
        prev = 1.0
        for rank_i in range(n_tests - 1, -1, -1):
            orig_i = sorted_idx[rank_i]
            bh_adj[orig_i] = min(bh_adj[orig_i], prev)
            prev = bh_adj[orig_i]
        bh_adj = np.minimum(bh_adj, 1.0)
        for i, k in enumerate(keys):
            results[k]["mann_whitney_p_bh"] = round(float(bh_adj[i]), 6)
            results[k]["significant_bh_05"] = float(bh_adj[i]) < 0.05

    return results


# ── 2b. Token Consumption Statistical Tests ──────────────────────────────────


def _interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d magnitude."""
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    elif ad < 0.5:
        return "small"
    elif ad < 0.8:
        return "medium"
    else:
        return "large"


def get_tokens_per_condition(conn) -> dict[int, list[int]]:
    """Return {condition_id: [total_tokens, ...]} for all episodes."""
    tokens = {cid: [] for cid in range(1, 5)}
    rows = conn.execute(
        "SELECT condition_id, total_tokens FROM episodes "
        "WHERE total_tokens IS NOT NULL ORDER BY condition_id, task_order"
    ).fetchall()
    for r in rows:
        tokens[r["condition_id"]].append(r["total_tokens"])
    return tokens


def run_token_statistical_tests(conn) -> dict:
    """Comprehensive statistical analysis of token consumption across conditions.

    Produces:
      - Bootstrap 95% CIs for per-condition means (10,000 resamples)
      - Bootstrap 95% CIs for all pairwise mean differences
      - Cohen's d for all 6 pairwise comparisons with interpretation
      - Mann-Whitney U for all 6 pairs
      - Kruskal-Wallis H across all 4 conditions
      - Bonferroni correction for 6 multiple comparisons
    """
    if not HAS_SCIPY:
        return {"error": "scipy not installed — cannot run token statistical tests"}

    tokens = get_tokens_per_condition(conn)

    # Check minimum data requirements
    for cid in range(1, 5):
        if len(tokens[cid]) < 2:
            return {"error": f"Not enough data for C{cid} ({len(tokens[cid])} episodes)"}

    rng = np.random.default_rng(42)
    results = {}

    # ── Per-condition bootstrap CIs for mean token consumption ──
    condition_stats = {}
    for cid in range(1, 5):
        data = tokens[cid]
        arr = np.asarray(data, dtype=np.float64)
        boot_mean, ci_lo, ci_hi = bayesian_bootstrap_mean(data, n_boot=10000, rng=rng)
        condition_stats[f"C{cid}"] = {
            "n": len(data),
            "mean": round(float(np.mean(arr)), 2),
            "median": round(float(np.median(arr)), 2),
            "std": round(float(np.std(arr, ddof=1)), 2),
            "bootstrap_mean": round(boot_mean, 2),
            "ci_95": [round(ci_lo, 2), round(ci_hi, 2)],
        }
    results["per_condition"] = condition_stats

    # ── Kruskal-Wallis H test across all 4 conditions ──
    h_stat, h_p = sp_stats.kruskal(tokens[1], tokens[2], tokens[3], tokens[4])
    results["kruskal_wallis"] = {
        "H_statistic": round(float(h_stat), 4),
        "p_value": round(float(h_p), 8),
        "significant_05": float(h_p) < 0.05,
        "df": 3,
        "interpretation": (
            "Significant difference across conditions"
            if float(h_p) < 0.05
            else "No significant difference across conditions"
        ),
    }

    # ── All 6 pairwise comparisons ──
    pairs = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    pairwise = {}
    raw_p_values = []
    pair_keys = []

    for ca, cb in pairs:
        key = f"C{ca}_vs_C{cb}"
        pair_keys.append(key)

        data_a = tokens[ca]
        data_b = tokens[cb]

        # Bootstrap difference in means (B - A)
        diff_mean, diff_lo, diff_hi, prob_b_better = bayesian_bootstrap_diff(
            data_a, data_b, n_boot=10000, rng=rng
        )

        # Mann-Whitney U
        u_stat, u_p = sp_stats.mannwhitneyu(data_a, data_b, alternative="two-sided")
        raw_p_values.append(float(u_p))

        # Cohen's d
        d = cohens_d(data_a, data_b)
        d_interp = _interpret_cohens_d(d) if d is not None else None

        # Percent difference relative to A
        mean_a = float(np.mean(data_a))
        mean_b = float(np.mean(data_b))
        pct_diff = round((mean_b - mean_a) / mean_a * 100, 2) if mean_a > 0 else None

        pairwise[key] = {
            "n_a": len(data_a),
            "n_b": len(data_b),
            "mean_a": round(mean_a, 2),
            "mean_b": round(mean_b, 2),
            "diff_mean": round(diff_mean, 2),
            "diff_ci_95": [round(diff_lo, 2), round(diff_hi, 2)],
            "pct_diff": pct_diff,
            "prob_b_lower": round(1 - prob_b_better, 4),  # P(B uses fewer tokens)
            "mann_whitney_U": round(float(u_stat), 2),
            "mann_whitney_p": round(float(u_p), 8),
            "cohens_d": round(d, 4) if d is not None else None,
            "effect_size": d_interp,
        }

    # ── Bonferroni correction (6 tests) ──
    n_tests = len(raw_p_values)
    for i, key in enumerate(pair_keys):
        p_adj = min(raw_p_values[i] * n_tests, 1.0)
        pairwise[key]["mann_whitney_p_bonferroni"] = round(p_adj, 8)
        pairwise[key]["significant_bonferroni_05"] = p_adj < 0.05

    # ── Benjamini-Hochberg correction ──
    sorted_idx = np.argsort(raw_p_values)
    bh_adj = np.zeros(n_tests)
    for rank_i, orig_i in enumerate(sorted_idx):
        bh_adj[orig_i] = raw_p_values[orig_i] * n_tests / (rank_i + 1)
    # Enforce monotonicity
    prev = 1.0
    for rank_i in range(n_tests - 1, -1, -1):
        orig_i = sorted_idx[rank_i]
        bh_adj[orig_i] = min(bh_adj[orig_i], prev)
        prev = bh_adj[orig_i]
    bh_adj = np.minimum(bh_adj, 1.0)
    for i, key in enumerate(pair_keys):
        pairwise[key]["mann_whitney_p_bh"] = round(float(bh_adj[i]), 8)
        pairwise[key]["significant_bh_05"] = float(bh_adj[i]) < 0.05

    results["pairwise"] = pairwise

    return results


def print_token_statistical_results(stats: dict):
    """Print token consumption statistical test results to console."""
    print("=" * 100)
    print("TOKEN CONSUMPTION — STATISTICAL TESTS")
    print("=" * 100)

    if "error" in stats:
        print(f"  {stats['error']}")
        return

    # Per-condition summary
    print("\n  Per-Condition Bootstrap 95% CI (mean tokens):")
    print("  " + "-" * 80)
    print(f"  {'Condition':<16} {'N':>6} {'Mean':>10} {'Median':>10} {'Std':>10} {'95% CI':>24}")
    print("  " + "-" * 80)
    for key in ["C1", "C2", "C3", "C4"]:
        s = stats["per_condition"][key]
        ci = s["ci_95"]
        print(
            f"  {key:<16} {s['n']:>6} {s['mean']:>10.0f} {s['median']:>10.0f} "
            f"{s['std']:>10.0f} [{ci[0]:>10.0f}, {ci[1]:>10.0f}]"
        )

    # Kruskal-Wallis
    kw = stats["kruskal_wallis"]
    print(f"\n  Kruskal-Wallis H test (df={kw['df']}):")
    print(f"    H = {kw['H_statistic']:.4f},  p = {kw['p_value']:.8f}")
    sig = "***" if kw["significant_05"] else "n.s."
    print(f"    {kw['interpretation']}  {sig}")

    # Pairwise comparisons
    print("\n  Pairwise Comparisons (6 tests, Bonferroni-corrected):")
    print("  " + "-" * 96)
    print(
        f"  {'Pair':<12} {'Diff':>8} {'% Diff':>8} {'95% CI':>24} "
        f"{'U':>10} {'p-raw':>12} {'p-Bonf':>12} {'d':>8} {'Effect':>12}"
    )
    print("  " + "-" * 96)

    for key in ["C1_vs_C2", "C1_vs_C3", "C1_vs_C4", "C2_vs_C3", "C2_vs_C4", "C3_vs_C4"]:
        p = stats["pairwise"][key]
        ci = p["diff_ci_95"]
        sig_marker = " ***" if p["significant_bonferroni_05"] else "    "
        pct = f"{p['pct_diff']:+.1f}%" if p["pct_diff"] is not None else "N/A"
        d_str = f"{p['cohens_d']:.4f}" if p["cohens_d"] is not None else "N/A"
        eff = p["effect_size"] or "N/A"
        print(
            f"  {key:<12} {p['diff_mean']:>+8.0f} {pct:>8} "
            f"[{ci[0]:>+10.0f}, {ci[1]:>+10.0f}] "
            f"{p['mann_whitney_U']:>10.0f} {p['mann_whitney_p']:>12.8f} "
            f"{p['mann_whitney_p_bonferroni']:>12.8f}{sig_marker} "
            f"{d_str:>8} {eff:>12}"
        )

    print()


# ── 3. Convergence Analysis ─────────────────────────────────────────────────


def compute_convergence(conn, stability_window=20) -> dict:
    """For each bandit condition (C2-C4): when does the best arm stabilize?

    Definition: same arm is best for the last N consecutive episodes.
    Also computes exploration ratio over time.
    """
    results = {}

    for cid in [2, 3, 4]:
        # Replay the bandit history
        rows = conn.execute(
            f"""
            SELECT e.task_order, e.preset_id, {_reward_sql()} as reward
            FROM episodes e
            LEFT JOIN feedback f ON e.episode_id = f.episode_id
            WHERE e.condition_id = ? AND e.preset_id IS NOT NULL
            ORDER BY e.task_order
        """,
            (cid,),
        ).fetchall()

        if not rows:
            results[COND_SHORT[cid]] = {"stabilized_at": None, "best_arm": None}
            continue

        presets = list(set(r["preset_id"] for r in rows))
        alphas = {p: 1.0 for p in presets}
        betas_ = {p: 1.0 for p in presets}

        best_arm_seq = []
        arm_means_over_time = {p: [] for p in presets}
        best_mean_over_time = []
        exploration_ratio_over_time = []

        pull_counts = {p: 0 for p in presets}

        for row in rows:
            preset = row["preset_id"]
            reward = row["reward"]

            pull_counts[preset] = pull_counts.get(preset, 0) + 1

            if reward is not None and preset in alphas:
                alphas[preset] += reward
                betas_[preset] += 1.0 - reward

            means = {p: alphas[p] / (alphas[p] + betas_[p]) for p in presets}
            current_best = max(means, key=means.get)
            best_arm_seq.append(current_best)

            for p in presets:
                arm_means_over_time[p].append(
                    {
                        "iter": row["task_order"],
                        "mean": round(means[p], 4),
                    }
                )

            best_mean_over_time.append(
                {
                    "iter": row["task_order"],
                    "mean": round(means[current_best], 4),
                    "arm": current_best,
                }
            )

            # Exploration ratio at this point
            total_pulls = sum(pull_counts.values())
            max_pulls = max(pull_counts.values())
            expl = (total_pulls - max_pulls) / total_pulls if total_pulls > 0 else 1.0
            exploration_ratio_over_time.append(
                {
                    "iter": row["task_order"],
                    "ratio": round(expl, 4),
                }
            )

        # Find stabilization: when did the best arm stop changing?
        stabilized_at = None
        final_best = best_arm_seq[-1] if best_arm_seq else None
        if len(best_arm_seq) >= stability_window:
            for i in range(len(best_arm_seq) - stability_window, -1, -1):
                window = best_arm_seq[i : i + stability_window]
                if all(a == window[0] for a in window):
                    if window[0] == final_best:
                        stabilized_at = i + 1  # 1-indexed position where stability begins
                    break

        results[COND_SHORT[cid]] = {
            "stabilized_at": stabilized_at,
            "total_episodes": len(best_arm_seq),
            "best_arm": final_best,
            "stability_window": stability_window,
            "best_mean_history": best_mean_over_time,
            "arm_histories": arm_means_over_time,
            "exploration_ratio": exploration_ratio_over_time,
        }

    return results


# ── 4. Theme Breakdown ──────────────────────────────────────────────────────


def compute_theme_breakdown(conn) -> dict:
    """Per-theme x per-condition reward matrix."""
    themes = get_themes(conn)
    matrix = {}  # theme -> {cid -> mean_reward}
    theme_variance = {}

    for theme in themes:
        tids = [
            r["task_id"]
            for r in conn.execute("SELECT task_id FROM tasks WHERE theme = ?", (theme,)).fetchall()
        ]
        if not tids:
            continue

        ph = ",".join("?" * len(tids))
        theme_rewards = {}

        for cid in range(1, 5):
            params = tids + [cid]
            rows = conn.execute(
                f"""
                SELECT {_reward_sql()} as reward
                FROM feedback f
                JOIN episodes e ON f.episode_id = e.episode_id
                WHERE e.task_id IN ({ph}) AND e.condition_id = ?
            """,
                params,
            ).fetchall()
            rews = [r["reward"] for r in rows if r["reward"] is not None]
            theme_rewards[cid] = round(float(np.mean(rews)), 4) if rews else None

        matrix[theme] = theme_rewards

        # Variance across conditions for this theme
        vals = [v for v in theme_rewards.values() if v is not None]
        theme_variance[theme] = round(float(np.var(vals)), 6) if len(vals) > 1 else 0.0

    # Per-theme best preset (for conditions with bandits)
    theme_presets = {}
    for theme in themes:
        tids = [
            r["task_id"]
            for r in conn.execute("SELECT task_id FROM tasks WHERE theme = ?", (theme,)).fetchall()
        ]
        if not tids:
            continue
        ph = ",".join("?" * len(tids))
        # Most-used preset per condition for this theme's tasks
        theme_presets[theme] = {}
        for cid in [2, 3, 4]:
            params = tids + [cid]
            best_preset = conn.execute(
                f"""
                SELECT preset_id, COUNT(*) as cnt
                FROM episodes
                WHERE task_id IN ({ph}) AND condition_id = ?
                  AND preset_id IS NOT NULL
                GROUP BY preset_id ORDER BY cnt DESC LIMIT 1
            """,
                params,
            ).fetchone()
            theme_presets[theme][cid] = best_preset["preset_id"] if best_preset else None

    # Rank themes by variance (which themes drive the most difference)
    sorted_themes = sorted(theme_variance.items(), key=lambda x: x[1], reverse=True)

    return {
        "matrix": matrix,
        "theme_variance": dict(sorted_themes),
        "theme_presets": theme_presets,
        "themes": themes,
    }


# ── 5. Feedback Modality Comparison ─────────────────────────────────────────


def compute_modality_comparison(conn) -> dict:
    """Compare C2 vs C3 (embedding value) and C2/C3 vs C4 (structured vs qualitative)."""
    rewards = get_rewards_per_condition(conn)

    result = {}

    # C2 vs C3: Does the explanation embedding add value?
    c2, c3 = rewards[2], rewards[3]
    c2c3 = {
        "c2_mean_reward": round(float(np.mean(c2)), 4) if c2 else None,
        "c3_mean_reward": round(float(np.mean(c3)), 4) if c3 else None,
        "c2_n": len(c2),
        "c3_n": len(c3),
    }
    if c2 and c3 and HAS_SCIPY:
        u, p = sp_stats.mannwhitneyu(c2, c3, alternative="two-sided")
        c2c3["mann_whitney_p"] = round(float(p), 6)
        c2c3["cohens_d"] = round(cohens_d(c2, c3), 4) if cohens_d(c2, c3) is not None else None

    # Convergence speed: which stabilized first?
    convergence = compute_convergence(conn)
    c2c3["c2_stabilized_at"] = convergence.get("C2", {}).get("stabilized_at")
    c2c3["c3_stabilized_at"] = convergence.get("C3", {}).get("stabilized_at")

    # Embedding count check (C3 differentiator)
    c3_emb = conn.execute(
        "SELECT COUNT(*) FROM feedback_embeddings fe "
        "JOIN feedback f ON fe.feedback_id = f.feedback_id "
        "JOIN episodes e ON f.episode_id = e.episode_id WHERE e.condition_id = 3"
    ).fetchone()[0]
    c2_emb = conn.execute(
        "SELECT COUNT(*) FROM feedback_embeddings fe "
        "JOIN feedback f ON fe.feedback_id = f.feedback_id "
        "JOIN episodes e ON f.episode_id = e.episode_id WHERE e.condition_id = 2"
    ).fetchone()[0]
    c2c3["c3_embeddings_stored"] = c3_emb
    c2c3["c2_embeddings_stored"] = c2_emb
    c2c3["embedding_differentiator"] = c3_emb > 0 and c2_emb == 0

    result["c2_vs_c3"] = c2c3

    # Structured (C2+C3) vs Qualitative (C4)
    c4 = rewards[4]
    structured = c2 + c3  # Pool C2 and C3
    struct_vs_qual = {
        "structured_mean": round(float(np.mean(structured)), 4) if structured else None,
        "qualitative_mean": round(float(np.mean(c4)), 4) if c4 else None,
        "structured_n": len(structured),
        "qualitative_n": len(c4),
    }
    if structured and c4 and HAS_SCIPY:
        u, p = sp_stats.mannwhitneyu(structured, c4, alternative="two-sided")
        struct_vs_qual["mann_whitney_p"] = round(float(p), 6)
        d = cohens_d(structured, c4)
        struct_vs_qual["cohens_d"] = round(d, 4) if d is not None else None

    result["structured_vs_qualitative"] = struct_vs_qual

    # Parse rate per modality
    for cid in [2, 3, 4]:
        n = conn.execute("SELECT COUNT(*) FROM episodes WHERE condition_id = ?", (cid,)).fetchone()[
            0
        ]
        fb = conn.execute(
            "SELECT COUNT(*) FROM feedback f JOIN episodes e ON f.episode_id = e.episode_id "
            "WHERE e.condition_id = ?",
            (cid,),
        ).fetchone()[0]
        result[f"parse_rate_c{cid}"] = round(fb / n * 100, 1) if n > 0 else 0.0

    # Parse rate trend per condition (rolling window=10)
    for cid in [2, 3, 4]:
        rows = conn.execute(
            """
            SELECT e.episode_id,
                   CASE WHEN f.feedback_id IS NOT NULL THEN 1 ELSE 0 END as has_fb
            FROM episodes e
            LEFT JOIN feedback f ON e.episode_id = f.episode_id
            WHERE e.condition_id = ?
            ORDER BY e.task_order
        """,
            (cid,),
        ).fetchall()
        flags = [r["has_fb"] for r in rows]
        trend = []
        window = 10
        for i in range(len(flags)):
            start = max(0, i - window + 1)
            chunk = flags[start : i + 1]
            trend.append(round(sum(chunk) / len(chunk) * 100, 1))
        result[f"parse_rate_trend_c{cid}"] = trend

    return result


# ── 6. Ground Truth Hit Rate ───────────────────────────────────────────────


def compute_ground_truth_hit_rate(conn) -> dict:
    """Compute ground truth skill retrieval rates per condition.

    For each episode whose task has ground_truth_skills, checks whether
    any of the top-5 retrieved skills match ground truth.
    """
    result = {}

    for cid in range(1, 5):
        rows = conn.execute(
            """
            SELECT e.episode_id, t.difficulty, t.theme,
                   MAX(CASE WHEN rr.is_ground_truth = 1 THEN 1 ELSE 0 END) as has_gt,
                   SUM(CASE WHEN rr.is_ground_truth = 1 THEN 1 ELSE 0 END) as gt_count
            FROM episodes e
            JOIN tasks t ON e.task_id = t.task_id
            LEFT JOIN retrieval_results rr ON e.episode_id = rr.episode_id AND rr.rank <= 5
            WHERE e.condition_id = ?
              AND t.ground_truth_skills IS NOT NULL
              AND t.ground_truth_skills != '[]'
            GROUP BY e.episode_id
        """,
            (cid,),
        ).fetchall()

        if not rows:
            result[cid] = {"hit_rate": None, "mean_gt_in_top5": None, "n": 0}
            continue

        hits = sum(1 for r in rows if r["has_gt"])
        hit_rate = hits / len(rows)
        mean_gt = float(np.mean([r["gt_count"] for r in rows]))

        # By difficulty
        by_diff = {}
        for diff in ["easy", "medium", "hard"]:
            diff_rows = [r for r in rows if r["difficulty"] == diff]
            if diff_rows:
                diff_hits = sum(1 for r in diff_rows if r["has_gt"])
                by_diff[diff] = {
                    "hit_rate": round(diff_hits / len(diff_rows), 4),
                    "n": len(diff_rows),
                }

        # By theme
        by_theme = {}
        themes = set(r["theme"] for r in rows)
        for theme in themes:
            theme_rows = [r for r in rows if r["theme"] == theme]
            if theme_rows:
                theme_hits = sum(1 for r in theme_rows if r["has_gt"])
                by_theme[theme] = {
                    "hit_rate": round(theme_hits / len(theme_rows), 4),
                    "n": len(theme_rows),
                }

        result[cid] = {
            "hit_rate": round(hit_rate, 4),
            "mean_gt_in_top5": round(mean_gt, 4),
            "n": len(rows),
            "hits": hits,
            "by_difficulty": by_diff,
            "by_theme": by_theme,
        }

    return result


def print_ground_truth(gt: dict):
    """Print ground truth hit rate results."""
    print("=" * 100)
    print("GROUND TRUTH HIT RATE (top-5 contains >= 1 GT skill)")
    print("=" * 100)

    headers = ["Condition", "N", "Hits", "Hit Rate", "Mean GT in Top-5"]
    widths = [18, 6, 6, 10, 16]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("-" * sum(widths))

    for cid in range(1, 5):
        g = gt.get(cid, {})
        if g.get("n", 0) == 0:
            print(f"  {COND_NAMES[cid]:18s}  {'0':6s}  {'---':6s}  {'N/A':10s}  {'N/A':16s}")
            continue
        vals = [
            COND_NAMES[cid],
            str(g["n"]),
            str(g["hits"]),
            f"{g['hit_rate']:.1%}",
            f"{g['mean_gt_in_top5']:.2f}",
        ]
        print("  ".join(v.ljust(w) for v, w in zip(vals, widths)))

    # By difficulty
    print("\n  By difficulty:")
    for cid in range(1, 5):
        g = gt.get(cid, {})
        parts = []
        for diff in ["easy", "medium", "hard"]:
            d = g.get("by_difficulty", {}).get(diff, {})
            if d:
                parts.append(f"{diff}={d['hit_rate']:.1%}(n={d['n']})")
        if parts:
            print(f"    {COND_SHORT[cid]}: {', '.join(parts)}")

    print()


# ── 7. Jaccard Similarity Tracking ────────────────────────────────────────


def compute_jaccard_trajectories(conn) -> dict:
    """Compute pairwise Jaccard similarity of top-5 skill sets across conditions.

    For each task, compares what skills each condition retrieved in its top-5.
    Returns per-task and rolling trajectories.
    """
    # Get all task_orders that have episodes in all 4 conditions
    task_orders = conn.execute("""
        SELECT task_order FROM episodes
        GROUP BY task_order
        HAVING COUNT(DISTINCT condition_id) = 4
        ORDER BY task_order
    """).fetchall()
    task_orders = [r["task_order"] for r in task_orders]

    if not task_orders:
        return {"pairs": {}, "task_orders": [], "milestones": {}}

    # Build skill sets per (task_order, condition_id)
    skill_sets = {}
    for to in task_orders:
        for cid in range(1, 5):
            rows = conn.execute(
                """
                SELECT rr.skill_id FROM retrieval_results rr
                JOIN episodes e ON rr.episode_id = e.episode_id
                WHERE e.task_order = ? AND e.condition_id = ? AND rr.rank <= 5
            """,
                (to, cid),
            ).fetchall()
            skill_sets[(to, cid)] = set(r["skill_id"] for r in rows)

    # Compute pairwise Jaccard for all 6 pairs
    pairs = [
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 3),
        (2, 4),
        (3, 4),
    ]
    trajectories = {}
    for a, b in pairs:
        key = f"C{a}_vs_C{b}"
        jaccards = []
        for to in task_orders:
            sa = skill_sets.get((to, a), set())
            sb = skill_sets.get((to, b), set())
            union = sa | sb
            if union:
                j = len(sa & sb) / len(union)
            else:
                j = 1.0
            jaccards.append({"task_order": to, "jaccard": round(j, 4)})

        # Rolling mean (window=30)
        raw = [j["jaccard"] for j in jaccards]
        window = min(30, len(raw))
        rolling = []
        for i in range(len(raw)):
            start = max(0, i - window + 1)
            rolling.append(round(float(np.mean(raw[start : i + 1])), 4))

        overall = round(float(np.mean(raw)), 4) if raw else None
        perfect_overlap = sum(1 for j in raw if j == 1.0)

        trajectories[key] = {
            "points": jaccards,
            "rolling": rolling,
            "overall_mean": overall,
            "perfect_overlap_pct": round(perfect_overlap / len(raw) * 100, 1) if raw else 0,
        }

    # Milestones
    milestones = {}
    milestone_points = [10, 50, 100, 150, 200, 250, 300]
    for a, b in pairs:
        key = f"C{a}_vs_C{b}"
        traj = trajectories[key]["points"]
        milestone_vals = {}
        for mp in milestone_points:
            subset = [j["jaccard"] for j in traj if j["task_order"] <= mp]
            if subset:
                milestone_vals[mp] = round(float(np.mean(subset)), 4)
        milestones[key] = milestone_vals

    return {
        "pairs": trajectories,
        "task_orders": task_orders,
        "milestones": milestones,
    }


def print_jaccard(jaccard: dict):
    """Print Jaccard similarity results."""
    print("=" * 100)
    print("JACCARD SIMILARITY (top-5 skill overlap between conditions)")
    print("=" * 100)

    pairs = jaccard.get("pairs", {})
    if not pairs:
        print("  No data available.")
        print()
        return

    headers = ["Pair", "Overall", "Perfect%"]
    widths = [14, 10, 10]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("-" * sum(widths))

    for key in sorted(pairs.keys()):
        p = pairs[key]
        vals = [
            key,
            f"{p['overall_mean']:.4f}" if p["overall_mean"] is not None else "N/A",
            f"{p['perfect_overlap_pct']:.1f}%",
        ]
        print("  ".join(v.ljust(w) for v, w in zip(vals, widths)))

    # Milestones for key pair (C1 vs C4)
    ms = jaccard.get("milestones", {}).get("C1_vs_C4", {})
    if ms:
        print("\n  C1 vs C4 milestones: ", end="")
        print(", ".join(f"@{k}={v:.3f}" for k, v in sorted(ms.items())))

    print()


# ── 8. Token Input/Output Breakdown ───────────────────────────────────────


def compute_token_breakdown(conn) -> dict:
    """Break down token usage by input vs output per condition and theme."""
    result = {}

    for cid in range(1, 5):
        row = conn.execute(
            """
            SELECT COUNT(*) as n,
                   AVG(input_tokens) as mean_input,
                   AVG(output_tokens) as mean_output,
                   AVG(total_tokens) as mean_total,
                   SUM(input_tokens) as sum_input,
                   SUM(output_tokens) as sum_output,
                   SUM(total_tokens) as sum_total
            FROM episodes WHERE condition_id = ?
        """,
            (cid,),
        ).fetchone()

        n = row["n"] or 0
        if n == 0:
            result[cid] = {"n": 0}
            continue

        mean_total = row["mean_total"] or 1
        input_pct = (row["mean_input"] or 0) / mean_total * 100
        output_pct = (row["mean_output"] or 0) / mean_total * 100

        # Per-theme breakdown
        by_theme = {}
        theme_rows = conn.execute(
            """
            SELECT t.theme,
                   COUNT(*) as n,
                   AVG(e.input_tokens) as mean_input,
                   AVG(e.output_tokens) as mean_output,
                   AVG(e.total_tokens) as mean_total
            FROM episodes e
            JOIN tasks t ON e.task_id = t.task_id
            WHERE e.condition_id = ?
            GROUP BY t.theme
        """,
            (cid,),
        ).fetchall()
        for tr in theme_rows:
            by_theme[tr["theme"]] = {
                "n": tr["n"],
                "mean_input": round(tr["mean_input"]) if tr["mean_input"] else 0,
                "mean_output": round(tr["mean_output"]) if tr["mean_output"] else 0,
                "mean_total": round(tr["mean_total"]) if tr["mean_total"] else 0,
            }

        result[cid] = {
            "n": n,
            "mean_input": round(row["mean_input"]) if row["mean_input"] else 0,
            "mean_output": round(row["mean_output"]) if row["mean_output"] else 0,
            "mean_total": round(row["mean_total"]) if row["mean_total"] else 0,
            "sum_input": row["sum_input"] or 0,
            "sum_output": row["sum_output"] or 0,
            "sum_total": row["sum_total"] or 0,
            "input_pct": round(input_pct, 1),
            "output_pct": round(output_pct, 1),
            "by_theme": by_theme,
        }

    return result


def print_token_breakdown(tokens: dict):
    """Print token input/output breakdown."""
    print("=" * 100)
    print("TOKEN BREAKDOWN (input vs output per condition)")
    print("=" * 100)

    headers = ["Condition", "N", "Mean In", "Mean Out", "Mean Total", "In%", "Out%", "Sum Total"]
    widths = [18, 5, 9, 9, 10, 6, 6, 12]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("-" * sum(widths))

    for cid in range(1, 5):
        t = tokens.get(cid, {})
        if t.get("n", 0) == 0:
            continue
        vals = [
            COND_NAMES[cid],
            str(t["n"]),
            f"{t['mean_input']:,}",
            f"{t['mean_output']:,}",
            f"{t['mean_total']:,}",
            f"{t['input_pct']:.0f}%",
            f"{t['output_pct']:.0f}%",
            f"{t['sum_total']:,}",
        ]
        print("  ".join(v.ljust(w) for v, w in zip(vals, widths)))

    # Per-theme for C1 vs C4 (the key comparison)
    c1_themes = tokens.get(1, {}).get("by_theme", {})
    c4_themes = tokens.get(4, {}).get("by_theme", {})
    if c1_themes and c4_themes:
        print("\n  C1 vs C4 per theme (mean total tokens):")
        for theme in sorted(c1_themes.keys()):
            c1v = c1_themes.get(theme, {}).get("mean_total", 0)
            c4v = c4_themes.get(theme, {}).get("mean_total", 0)
            diff_pct = ((c4v - c1v) / c1v * 100) if c1v > 0 else 0
            print(f"    {theme[:28]:28s}  C1={c1v:>7,}  C4={c4v:>7,}  ({diff_pct:+.1f}%)")

    print()


# ── 9. Anchor Spread Validation (C4) ──────────────────────────────────────


def compute_anchor_validation(conn) -> dict:
    """Validate C4 anchor-inferred scores are well-distributed, not clustered near 0.5."""
    rows = conn.execute("""
        SELECT f.inferred_recency, f.inferred_importance, f.inferred_relevance
        FROM feedback f
        JOIN episodes e ON f.episode_id = e.episode_id
        WHERE e.condition_id = 4
          AND f.inferred_recency IS NOT NULL
    """).fetchall()

    if not rows:
        return {"n": 0}

    result = {"n": len(rows), "dimensions": {}}

    for dim in ["recency", "importance", "relevance"]:
        scores = [r[f"inferred_{dim}"] for r in rows if r[f"inferred_{dim}"] is not None]
        if not scores:
            continue
        arr = np.array(scores)
        near_center = sum(1 for s in scores if 0.4 <= s <= 0.6)
        result["dimensions"][dim] = {
            "mean": round(float(np.mean(arr)), 4),
            "std": round(float(np.std(arr)), 4),
            "min": round(float(np.min(arr)), 4),
            "max": round(float(np.max(arr)), 4),
            "spread": round(float(np.max(arr) - np.min(arr)), 4),
            "q25": round(float(np.percentile(arr, 25)), 4),
            "q50": round(float(np.percentile(arr, 50)), 4),
            "q75": round(float(np.percentile(arr, 75)), 4),
            "pct_near_center": round(near_center / len(scores) * 100, 1),
        }

    return result


def print_anchor_validation(anchor: dict):
    """Print anchor spread validation results."""
    print("=" * 100)
    print("C4 ANCHOR SPREAD VALIDATION")
    print("=" * 100)

    if anchor.get("n", 0) == 0:
        print("  No C4 feedback data available.")
        print()
        return

    print(f"  C4 episodes with anchor-inferred scores: {anchor['n']}\n")

    headers = ["Dimension", "Mean", "Std", "Min", "Max", "Spread", "Q25", "Q50", "Q75", "Near 0.5"]
    widths = [12, 7, 7, 7, 7, 7, 7, 7, 7, 8]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("-" * sum(widths))

    for dim in ["recency", "importance", "relevance"]:
        d = anchor.get("dimensions", {}).get(dim, {})
        if not d:
            continue
        vals = [
            dim.capitalize(),
            f"{d['mean']:.4f}",
            f"{d['std']:.4f}",
            f"{d['min']:.4f}",
            f"{d['max']:.4f}",
            f"{d['spread']:.4f}",
            f"{d['q25']:.4f}",
            f"{d['q50']:.4f}",
            f"{d['q75']:.4f}",
            f"{d['pct_near_center']:.0f}%",
        ]
        print("  ".join(v.ljust(w) for v, w in zip(vals, widths)))

    # Assessment
    dims = anchor.get("dimensions", {})
    all_spread = [dims[d]["spread"] for d in dims if "spread" in dims[d]]
    if all_spread:
        avg_spread = np.mean(all_spread)
        status = "GOOD" if avg_spread > 0.4 else "WARN" if avg_spread > 0.2 else "FAIL"
        print(f"\n  Average spread: {avg_spread:.4f} [{status}] (target > 0.40)")

    print()


# ── 10. Per-Difficulty Analysis ───────────────────────────────────────────────


def compute_difficulty_breakdown(conn) -> dict:
    """Per-difficulty x per-condition analysis (v3: 50E/100M/150H)."""
    result = {}
    for diff in ["easy", "medium", "hard"]:
        diff_data = {}
        for cid in range(1, 5):
            row = conn.execute(
                f"""
                SELECT COUNT(*) as n,
                       AVG(e.total_tokens) as mean_tokens,
                       AVG(e.step_count) as mean_steps,
                       AVG({_reward_sql()}) as mean_reward
                FROM episodes e
                JOIN tasks t ON e.task_id = t.task_id
                LEFT JOIN feedback f ON e.episode_id = f.episode_id
                WHERE e.condition_id = ? AND t.difficulty = ?
            """,
                (cid, diff),
            ).fetchone()
            n = row["n"] or 0
            diff_data[cid] = {
                "n": n,
                "mean_tokens": round(row["mean_tokens"]) if row["mean_tokens"] else None,
                "mean_steps": round(row["mean_steps"], 2) if row["mean_steps"] else None,
                "mean_reward": round(row["mean_reward"], 4) if row["mean_reward"] else None,
            }
        result[diff] = diff_data
    return result


def print_difficulty_breakdown(diff_data: dict):
    """Print per-difficulty breakdown."""
    print("=" * 100)
    print("PER-DIFFICULTY BREAKDOWN (50 Easy / 100 Medium / 150 Hard)")
    print("=" * 100)

    for diff in ["easy", "medium", "hard"]:
        d = diff_data.get(diff, {})
        print(f"\n  {diff.upper()}:")
        headers = ["Condition", "N", "Mean Rew", "Mean Steps", "Mean Tokens"]
        widths = [18, 5, 10, 12, 12]
        print("  " + "  ".join(h.ljust(w) for h, w in zip(headers, widths)))
        print("  " + "-" * sum(widths))
        for cid in range(1, 5):
            c = d.get(cid, {})
            vals = [
                COND_NAMES[cid],
                str(c.get("n", 0)),
                f"{c['mean_reward']:.4f}" if c.get("mean_reward") is not None else "N/A",
                f"{c['mean_steps']:.2f}" if c.get("mean_steps") is not None else "N/A",
                f"{c['mean_tokens']:,.0f}" if c.get("mean_tokens") is not None else "N/A",
            ]
            print("  " + "  ".join(v.ljust(w) for v, w in zip(vals, widths)))

    print()


# ── 11. Per-Domain Analysis ──────────────────────────────────────────────────


def compute_domain_breakdown(conn) -> dict:
    """Per-domain x per-condition analysis (v3: 10 domains)."""
    domains = [
        r["domain"]
        for r in conn.execute("SELECT DISTINCT domain FROM skills ORDER BY domain").fetchall()
    ]

    result = {}
    for domain in domains:
        # Get all skills in this domain
        skill_ids = [
            r["skill_id"]
            for r in conn.execute(
                "SELECT skill_id FROM skills WHERE domain = ?", (domain,)
            ).fetchall()
        ]
        if not skill_ids:
            continue

        ph = ",".join("?" * len(skill_ids))
        domain_data = {"skill_count": len(skill_ids)}

        for cid in range(1, 5):
            # How many times were skills from this domain retrieved?
            params = skill_ids + [cid]
            retrieval_count = conn.execute(
                f"""
                SELECT COUNT(*) FROM retrieval_results rr
                JOIN episodes e ON rr.episode_id = e.episode_id
                WHERE rr.skill_id IN ({ph}) AND e.condition_id = ?
            """,
                params,
            ).fetchone()[0]

            # How many times in top-5?
            top5_count = conn.execute(
                f"""
                SELECT COUNT(*) FROM retrieval_results rr
                JOIN episodes e ON rr.episode_id = e.episode_id
                WHERE rr.skill_id IN ({ph}) AND e.condition_id = ? AND rr.rank <= 5
            """,
                params,
            ).fetchone()[0]

            n_episodes = conn.execute(
                "SELECT COUNT(*) FROM episodes WHERE condition_id = ?", (cid,)
            ).fetchone()[0]

            domain_data[cid] = {
                "retrieval_count": retrieval_count,
                "top5_count": top5_count,
                "retrieval_per_episode": round(retrieval_count / n_episodes, 2)
                if n_episodes
                else 0,
            }

        result[domain] = domain_data

    return {"domains": domains, "data": result}


def print_domain_breakdown(domain_data: dict):
    """Print per-domain breakdown."""
    print("=" * 100)
    print("PER-DOMAIN ANALYSIS (10 skill domains)")
    print("=" * 100)

    headers = ["Domain", "Skills", "C1 top5", "C2 top5", "C3 top5", "C4 top5"]
    widths = [20, 7, 9, 9, 9, 9]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("-" * sum(widths))

    for domain in domain_data["domains"]:
        d = domain_data["data"].get(domain, {})
        vals = [
            domain[:18],
            str(d.get("skill_count", 0)),
        ]
        for cid in range(1, 5):
            c = d.get(cid, {})
            vals.append(str(c.get("top5_count", 0)))
        print("  ".join(v.ljust(w) for v, w in zip(vals, widths)))

    print()


# ── 12. External Ground Truth Validation ─────────────────────────────────────


def load_external_ground_truth() -> dict:
    """Load ground_truth_v3.json and build lookup dict."""
    gt_path = ROOT / "data" / "ground_truth_v3.json"
    if not gt_path.exists():
        print(f"  [WARN] Ground truth file not found: {gt_path}")
        return {}
    with open(gt_path) as f:
        entries = json.load(f)
    return {
        entry["task_id"]: {
            "relevant": entry["relevant_skill_ids"],
            "primary": entry.get("primary_skill_id"),
        }
        for entry in entries
    }


def compute_graded_ndcg_mrr(conn, gt_lookup: dict, condition_id: int | None = None) -> dict:
    """NDCG@5 and MRR using external ground truth with graded relevance.

    Graded relevance: primary_skill = 2.0, other relevant = 1.0, else 0.0
    """
    query = "SELECT e.episode_id, e.task_id FROM episodes e"
    params = []
    if condition_id:
        query += " WHERE e.condition_id = ?"
        params.append(condition_id)

    episodes = conn.execute(query, params).fetchall()
    if not episodes:
        return {"ndcg5": None, "mrr": None, "count": 0}

    ndcg_scores, rr_scores = [], []

    for ep in episodes:
        gt = gt_lookup.get(ep["task_id"])
        if not gt or not gt["relevant"]:
            continue

        gt_set = set(gt["relevant"])
        primary = gt.get("primary")

        retrieved = conn.execute(
            "SELECT skill_id FROM retrieval_results WHERE episode_id = ? ORDER BY rank",
            (ep["episode_id"],),
        ).fetchall()
        if not retrieved:
            continue

        # Graded NDCG@5
        k = min(5, len(retrieved))
        dcg = 0.0
        for i in range(k):
            sid = retrieved[i]["skill_id"]
            if sid == primary:
                rel = 2.0
            elif sid in gt_set:
                rel = 1.0
            else:
                rel = 0.0
            dcg += rel / math.log2(i + 2)

        # Ideal DCG: primary first (rel=2), then others (rel=1)
        ideal_rels = (
            [2.0] + [1.0] * (len(gt["relevant"]) - 1) if primary else [1.0] * len(gt["relevant"])
        )
        idcg = 0.0
        for i in range(min(k, len(ideal_rels))):
            idcg += ideal_rels[i] / math.log2(i + 2)

        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

        # MRR (any relevant hit)
        rr = 0.0
        for i, r in enumerate(retrieved):
            if r["skill_id"] in gt_set:
                rr = 1.0 / (i + 1)
                break
        rr_scores.append(rr)

    return {
        "ndcg5": round(float(np.mean(ndcg_scores)), 4) if ndcg_scores else None,
        "mrr": round(float(np.mean(rr_scores)), 4) if rr_scores else None,
        "count": len(ndcg_scores),
    }


def compute_ground_truth_external(conn, gt_lookup: dict) -> dict:
    """Full ground truth analysis using external file with per-difficulty breakdown."""
    result = {}

    for cid in range(1, 5):
        ir = compute_graded_ndcg_mrr(conn, gt_lookup, condition_id=cid)

        # Per-difficulty NDCG/MRR
        by_diff = {}
        for diff in ["easy", "medium", "hard"]:
            task_ids = [
                r["task_id"]
                for r in conn.execute(
                    "SELECT task_id FROM tasks WHERE difficulty = ?", (diff,)
                ).fetchall()
            ]
            diff_episodes = conn.execute(
                f"""SELECT e.episode_id, e.task_id FROM episodes e
                    WHERE e.condition_id = ? AND e.task_id IN ({",".join("?" * len(task_ids))})""",
                [cid] + task_ids,
            ).fetchall()

            ndcg_scores, rr_scores = [], []
            for ep in diff_episodes:
                gt = gt_lookup.get(ep["task_id"])
                if not gt or not gt["relevant"]:
                    continue
                gt_set = set(gt["relevant"])
                primary = gt.get("primary")
                retrieved = conn.execute(
                    "SELECT skill_id FROM retrieval_results WHERE episode_id = ? ORDER BY rank",
                    (ep["episode_id"],),
                ).fetchall()
                if not retrieved:
                    continue

                k = min(5, len(retrieved))
                dcg = 0.0
                for i in range(k):
                    sid = retrieved[i]["skill_id"]
                    rel = 2.0 if sid == primary else (1.0 if sid in gt_set else 0.0)
                    dcg += rel / math.log2(i + 2)
                ideal_rels = (
                    ([2.0] + [1.0] * (len(gt["relevant"]) - 1))
                    if primary
                    else ([1.0] * len(gt["relevant"]))
                )
                idcg = sum(ideal_rels[i] / math.log2(i + 2) for i in range(min(k, len(ideal_rels))))
                ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)
                rr = 0.0
                for i, r in enumerate(retrieved):
                    if r["skill_id"] in gt_set:
                        rr = 1.0 / (i + 1)
                        break
                rr_scores.append(rr)

            by_diff[diff] = {
                "ndcg5": round(float(np.mean(ndcg_scores)), 4) if ndcg_scores else None,
                "mrr": round(float(np.mean(rr_scores)), 4) if rr_scores else None,
                "n": len(ndcg_scores),
            }

        result[cid] = {
            "ndcg5": ir["ndcg5"],
            "mrr": ir["mrr"],
            "count": ir["count"],
            "by_difficulty": by_diff,
        }

    return result


def print_ground_truth_external(gt_ext: dict):
    """Print external ground truth NDCG/MRR results."""
    print("=" * 100)
    print("GROUND TRUTH — GRADED NDCG@5 / MRR (from ground_truth_v3.json)")
    print("=" * 100)

    headers = ["Condition", "N", "NDCG@5", "MRR", "Easy NDCG", "Med NDCG", "Hard NDCG"]
    widths = [18, 5, 8, 8, 11, 11, 11]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("-" * sum(widths))

    for cid in range(1, 5):
        g = gt_ext.get(cid, {})
        by_d = g.get("by_difficulty", {})
        vals = [
            COND_NAMES[cid],
            str(g.get("count", 0)),
            f"{g['ndcg5']:.4f}" if g.get("ndcg5") is not None else "N/A",
            f"{g['mrr']:.4f}" if g.get("mrr") is not None else "N/A",
            f"{by_d.get('easy', {}).get('ndcg5', 'N/A')}",
            f"{by_d.get('medium', {}).get('ndcg5', 'N/A')}",
            f"{by_d.get('hard', {}).get('ndcg5', 'N/A')}",
        ]
        print("  ".join(str(v).ljust(w) for v, w in zip(vals, widths)))

    print()


# ── 13. LaTeX Table Output ───────────────────────────────────────────────────


def emit_latex_primary_table(metrics: list[dict], path: Path):
    """Write primary metrics as a LaTeX table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Primary metrics by condition (v3, 300 tasks $\times$ 4 conditions).}",
        r"\label{tab:primary-metrics}",
        r"\begin{tabular}{lrrrrrrl}",
        r"\toprule",
        r"Condition & N & Mean Reward & Steps & Tokens & NDCG@5 & MRR & Best Arm \\",
        r"\midrule",
    ]
    for m in metrics:
        reward = f"{m['mean_reward']:.4f}" if m["mean_reward"] is not None else "---"
        steps = f"{m['mean_steps']:.1f}" if m["mean_steps"] is not None else "---"
        tokens = f"{m['mean_tokens']:,.0f}" if m["mean_tokens"] is not None else "---"
        ndcg = f"{m['ndcg5']:.4f}" if m["ndcg5"] is not None else "---"
        mrr = f"{m['mrr']:.4f}" if m["mrr"] is not None else "---"
        arm = m["best_arm"] or "---"
        lines.append(
            f"  {m['condition']} & {m['episodes']} & {reward} & {steps} "
            f"& {tokens} & {ndcg} & {mrr} & {arm} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    path.write_text("\n".join(lines))


def emit_latex_statistical_table(stats: dict, path: Path):
    """Write statistical test results as a LaTeX table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Statistical comparisons vs.\ C1 Control.}",
        r"\label{tab:stat-tests}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Comparison & $\Delta$ Reward & 95\% CI & $P(\text{better})$"
        r" & Cohen's $d$ & MW $p$ (BH) \\",
        r"\midrule",
    ]
    for key in ["C2", "C3", "C4"]:
        s = stats.get(key)
        if not s or "error" in s:
            continue
        ci = s["diff_ci_95"]
        d_val = f"{s['cohens_d']:.3f}" if s.get("cohens_d") is not None else "---"
        p_bh = f"{s.get('mann_whitney_p_bh', s['mann_whitney_p']):.4f}"
        lines.append(
            f"  {key} vs C1 & {s['diff_mean']:+.4f} "
            f"& [{ci[0]:+.4f}, {ci[1]:+.4f}] "
            f"& {s['prob_treatment_better']:.3f} "
            f"& {d_val} & {p_bh} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    path.write_text("\n".join(lines))


def emit_latex_difficulty_table(diff_data: dict, path: Path):
    """Write per-difficulty breakdown as a LaTeX table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Mean reward by difficulty level and condition.}",
        r"\label{tab:difficulty}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Difficulty & Metric & C1 & C2 & C3 & C4 \\",
        r"\midrule",
    ]
    for diff in ["easy", "medium", "hard"]:
        d = diff_data.get(diff, {})
        rewards = []
        tokens = []
        for cid in range(1, 5):
            c = d.get(cid, {})
            r = c.get("mean_reward")
            rewards.append(f"{r:.4f}" if r is not None else "---")
            t = c.get("mean_tokens")
            tokens.append(f"{t:,.0f}" if t is not None else "---")
        lines.append(f"  {diff.capitalize()} & Reward & {' & '.join(rewards)} \\\\")
        lines.append(f"   & Tokens & {' & '.join(tokens)} \\\\")
        if diff != "hard":
            lines.append(r"\addlinespace")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    path.write_text("\n".join(lines))


def emit_latex_jaccard_table(jaccard: dict, path: Path):
    """Write Jaccard similarity summary as a LaTeX table."""
    pairs = jaccard.get("pairs", {})
    if not pairs:
        return
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Pairwise Jaccard similarity of top-5 retrieved skill sets (target $<0.60$).}",
        r"\label{tab:jaccard}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Pair & Mean Jaccard & Perfect Overlap \% & Status \\",
        r"\midrule",
    ]
    for key in sorted(pairs.keys()):
        p = pairs[key]
        mean_j = p.get("overall_mean")
        pct = p.get("perfect_overlap_pct", 0)
        status = r"$\checkmark$" if (mean_j is not None and mean_j < 0.60) else r"$\times$"
        j_str = f"{mean_j:.4f}" if mean_j is not None else "---"
        lines.append(f"  {key.replace('_', ' ')} & {j_str} & {pct:.1f}\\% & {status} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    path.write_text("\n".join(lines))


def emit_latex_graded_gt_table(gt_ext: dict, path: Path):
    """Write graded NDCG@5/MRR table from external ground truth as LaTeX."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Graded NDCG@5 and MRR against external ground truth"
        r" (primary skill $=2$, other relevant $=1$).}",
        r"\label{tab:graded-gt}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Condition & N & NDCG@5 & MRR & Easy NDCG & Hard NDCG \\",
        r"\midrule",
    ]
    for cid in range(1, 5):
        g = gt_ext.get(cid, {})
        by_d = g.get("by_difficulty", {})
        n = g.get("count", 0)
        ndcg = f"{g['ndcg5']:.4f}" if g.get("ndcg5") is not None else "---"
        mrr = f"{g['mrr']:.4f}" if g.get("mrr") is not None else "---"
        easy_n = f"{by_d.get('easy', {}).get('ndcg5', '---')}"
        if isinstance(easy_n, float):
            easy_n = f"{easy_n:.4f}"
        hard_n = f"{by_d.get('hard', {}).get('ndcg5', '---')}"
        if isinstance(hard_n, float):
            hard_n = f"{hard_n:.4f}"
        lines.append(f"  {COND_NAMES[cid]} & {n} & {ndcg} & {mrr} & {easy_n} & {hard_n} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    path.write_text("\n".join(lines))


# ── 14. Cumulative Regret ────────────────────────────────────────────────────


def compute_cumulative_regret_curves(conn) -> dict:
    """Cumulative regret per condition: sum of (1.0 - reward) over episodes."""
    curves = {}
    for cid in range(1, 5):
        rows = conn.execute(
            f"""
            SELECT e.task_order, {_reward_sql()} as reward
            FROM feedback f
            JOIN episodes e ON f.episode_id = e.episode_id
            WHERE e.condition_id = ?
            ORDER BY e.task_order
        """,
            (cid,),
        ).fetchall()

        regret = 0.0
        points = []
        for r in rows:
            reward = r["reward"] if r["reward"] is not None else 0.0
            regret += 1.0 - reward
            points.append({"iter": r["task_order"], "regret": round(regret, 3)})

        curves[cid] = {
            "total_regret": round(regret, 3),
            "points": points,
        }
    return curves


# ── Figures ──────────────────────────────────────────────────────────────────


def fig_reward_boxplot(conn, fig_dir: Path):
    """Fig 1: Reward comparison box plot (4 conditions)."""
    rewards = get_rewards_per_condition(conn)
    fig, ax = plt.subplots(figsize=(10, 6))

    data = [rewards[cid] for cid in range(1, 5)]
    labels = [COND_SHORT[cid] for cid in range(1, 5)]
    colors = [COND_COLORS[cid] for cid in range(1, 5)]

    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        medianprops=dict(color=DARK_FG, linewidth=2),
        whiskerprops=dict(color=DARK_FG),
        capprops=dict(color=DARK_FG),
        flierprops=dict(markeredgecolor=DARK_FG, markersize=4),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add means as diamonds
    means = [float(np.mean(d)) if d else 0 for d in data]
    ax.scatter(range(1, 5), means, marker="D", color="white", s=50, zorder=5, label="Mean")

    ax.set_title("Composite Reward by Condition", fontsize=14, fontweight="bold")
    ax.set_ylabel("Composite Reward [0, 1]")
    ax.set_xlabel("Condition")
    ax.legend(loc="upper right")

    # Add sample sizes
    for i, (cid, d) in enumerate(zip(range(1, 5), data)):
        ax.text(
            i + 1,
            ax.get_ylim()[0] - 0.02,
            f"n={len(d)}",
            ha="center",
            va="top",
            fontsize=9,
            color="#888888",
        )

    plt.tight_layout()
    fig.savefig(fig_dir / "fig1_reward_boxplot.png")
    plt.close(fig)
    print("  Saved: fig1_reward_boxplot.png")


def fig_convergence_curves(conn, fig_dir: Path):
    """Fig 2: Convergence curves (posterior mean over iterations, all conditions).

    With 12 arms, shows top-5 arms (by final posterior mean) prominently
    and remaining arms as faded thin lines to keep the figure readable.
    """
    convergence = compute_convergence(conn)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for idx, cid in enumerate([2, 3, 4]):
        ax = axes[idx]
        key = COND_SHORT[cid]
        data = convergence.get(key, {})
        arm_histories = data.get("arm_histories", {})

        if not arm_histories:
            ax.set_title(f"{COND_NAMES[cid]}\n(no data)", fontsize=12)
            continue

        # Rank arms by final posterior mean
        final_means = {}
        for preset_id, history in arm_histories.items():
            if history:
                final_means[preset_id] = history[-1]["mean"]
            else:
                final_means[preset_id] = 0.5
        ranked = sorted(final_means.items(), key=lambda x: x[1], reverse=True)
        top_5 = {p for p, _ in ranked[:5]}

        for preset_id, history in arm_histories.items():
            iters = [h["iter"] for h in history]
            means = [h["mean"] for h in history]
            if preset_id in top_5:
                ax.plot(iters, means, linewidth=2, alpha=0.9, label=preset_id)
            else:
                ax.plot(iters, means, linewidth=0.7, alpha=0.25, color="#666666")

        # Mark stabilization point
        stab = data.get("stabilized_at")
        if stab is not None:
            ax.axvline(
                x=stab, color="#ff6b6b", linestyle="--", alpha=0.8, label=f"Stabilized @ {stab}"
            )

        ax.set_title(f"{COND_NAMES[cid]}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Episode")
        if idx == 0:
            ax.set_ylabel("Posterior Mean")
        ax.legend(fontsize=7, loc="lower right")
        ax.set_ylim(0.3, 0.8)

    fig.suptitle(
        "Bandit Convergence: Posterior Mean per Arm (top-5 highlighted)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(fig_dir / "fig2_convergence_curves.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig2_convergence_curves.png")


def fig_theme_heatmap(conn, fig_dir: Path):
    """Fig 3: Per-theme heatmap (theme x condition reward matrix)."""
    breakdown = compute_theme_breakdown(conn)
    matrix = breakdown["matrix"]
    themes = breakdown["themes"]

    if not themes:
        print("  Skipped: fig3_theme_heatmap.png (no theme data)")
        return

    # Build numpy matrix
    n_themes = len(themes)
    n_conds = 4
    data = np.full((n_themes, n_conds), np.nan)
    for i, theme in enumerate(themes):
        for j, cid in enumerate(range(1, 5)):
            val = matrix.get(theme, {}).get(cid)
            if val is not None:
                data[i, j] = val

    fig, ax = plt.subplots(figsize=(10, max(6, n_themes * 0.8 + 2)))

    # Custom colormap: dark theme friendly
    cmap = plt.cm.RdYlGn
    cmap.set_bad(color=DARK_ACCENT)

    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0.0, vmax=1.0)

    # Labels
    ax.set_xticks(range(n_conds))
    ax.set_xticklabels([COND_SHORT[cid] for cid in range(1, 5)])
    ax.set_yticks(range(n_themes))
    # Truncate long theme names
    short_themes = [t[:25] + "..." if len(t) > 28 else t for t in themes]
    ax.set_yticklabels(short_themes)

    # Add text annotations
    for i in range(n_themes):
        for j in range(n_conds):
            val = data[i, j]
            if not np.isnan(val):
                text_color = "black" if val > 0.5 else "white"
                ax.text(
                    j,
                    i,
                    f"{val:.3f}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color=text_color,
                    fontweight="bold",
                )

    ax.set_title("Mean Reward: Theme x Condition", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Composite Reward", shrink=0.8)

    plt.tight_layout()
    fig.savefig(fig_dir / "fig3_theme_heatmap.png")
    plt.close(fig)
    print("  Saved: fig3_theme_heatmap.png")


def fig_step_efficiency(conn, fig_dir: Path):
    """Fig 4: Mean token usage per condition (bar chart)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    token_data = {}
    for cid in range(1, 5):
        rows = conn.execute(
            "SELECT total_tokens FROM episodes WHERE condition_id = ?", (cid,)
        ).fetchall()
        token_data[cid] = [r["total_tokens"] for r in rows if r["total_tokens"] is not None]

    labels = [COND_SHORT[cid] for cid in range(1, 5)]
    colors = [COND_COLORS[cid] for cid in range(1, 5)]

    means = [float(np.mean(token_data[cid])) if token_data[cid] else 0 for cid in range(1, 5)]
    stds = [float(np.std(token_data[cid])) if token_data[cid] else 0 for cid in range(1, 5)]

    bars = ax.bar(
        range(1, 5),
        means,
        yerr=stds,
        capsize=5,
        color=colors,
        alpha=0.7,
        edgecolor=DARK_FG,
        linewidth=0.5,
    )
    ax.set_xticks(range(1, 5))
    ax.set_xticklabels(labels)
    ax.set_title("Mean Token Usage by Condition", fontsize=13, fontweight="bold")
    ax.set_ylabel("Tokens per Episode")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))

    # Add value labels on bars
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 200,
            f"{mean:,.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=DARK_FG,
        )

    plt.tight_layout()
    fig.savefig(fig_dir / "fig4_step_efficiency.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig4_step_efficiency.png")


def fig_ground_truth_hit_rate(conn, fig_dir: Path):
    """Fig 6: Ground truth hit rate per condition (bar chart with difficulty breakdown)."""
    gt = compute_ground_truth_hit_rate(conn)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Overall hit rate per condition
    hit_rates = [gt.get(cid, {}).get("hit_rate", 0) or 0 for cid in range(1, 5)]
    colors = [COND_COLORS[cid] for cid in range(1, 5)]
    labels = [COND_SHORT[cid] for cid in range(1, 5)]

    bars = ax1.bar(
        range(1, 5), hit_rates, color=colors, alpha=0.7, edgecolor=DARK_FG, linewidth=0.5
    )
    for bar, rate in zip(bars, hit_rates):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{rate:.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
            color=DARK_FG,
        )
    ax1.set_xticks(range(1, 5))
    ax1.set_xticklabels(labels)
    ax1.set_title("Overall Hit Rate", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Hit Rate (>= 1 GT skill in top-5)")
    ax1.set_ylim(0, 1.05)

    # Panel 2: By difficulty grouped bars
    diffs = ["easy", "medium", "hard"]
    x = np.arange(len(diffs))
    width = 0.2
    for i, cid in enumerate(range(1, 5)):
        rates = []
        for diff in diffs:
            d = gt.get(cid, {}).get("by_difficulty", {}).get(diff, {})
            rates.append(d.get("hit_rate", 0) or 0)
        ax2.bar(
            x + i * width, rates, width, label=COND_SHORT[cid], color=COND_COLORS[cid], alpha=0.7
        )
    ax2.set_xticks(x + 1.5 * width)
    ax2.set_xticklabels([d.capitalize() for d in diffs])
    ax2.set_title("Hit Rate by Difficulty", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Hit Rate")
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=9)

    fig.suptitle("Ground Truth Retrieval", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(fig_dir / "fig6_ground_truth_hit_rate.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig6_ground_truth_hit_rate.png")


def fig_jaccard_trajectory(conn, fig_dir: Path):
    """Fig 7: Jaccard similarity trajectory (C1 vs C2/C3/C4 over tasks)."""
    jaccard = compute_jaccard_trajectories(conn)
    pairs = jaccard.get("pairs", {})

    if not pairs:
        print("  Skipped: fig7_jaccard_trajectory.png (no data)")
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot key pairs: C1 vs others (shows divergence from control)
    key_pairs = [
        ("C1_vs_C2", COND_COLORS[2]),
        ("C1_vs_C3", COND_COLORS[3]),
        ("C1_vs_C4", COND_COLORS[4]),
    ]
    for key, color in key_pairs:
        p = pairs.get(key)
        if not p or not p["rolling"]:
            continue
        task_orders = [pt["task_order"] for pt in p["points"]]
        ax.plot(
            task_orders,
            p["rolling"],
            color=color,
            linewidth=2,
            alpha=0.9,
            label=f"{key.replace('_', ' ')} (mean={p['overall_mean']:.3f})",
        )

    # Target line
    ax.axhline(y=0.60, color="#ff6b6b", linestyle="--", alpha=0.5, label="Target < 0.60")

    ax.set_title(
        "Skill Set Divergence from Control (Rolling Jaccard, window=30)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Task Order")
    ax.set_ylabel("Jaccard Similarity (1.0 = identical, 0.0 = no overlap)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10)

    plt.tight_layout()
    fig.savefig(fig_dir / "fig7_jaccard_trajectory.png")
    plt.close(fig)
    print("  Saved: fig7_jaccard_trajectory.png")


def fig_cumulative_regret(conn, fig_dir: Path):
    """Fig 5: Cumulative regret curves (all 4 conditions)."""
    regret_data = compute_cumulative_regret_curves(conn)

    fig, ax = plt.subplots(figsize=(12, 7))

    for cid in range(1, 5):
        points = regret_data[cid]["points"]
        if not points:
            continue
        iters = [p["iter"] for p in points]
        regrets = [p["regret"] for p in points]
        ax.plot(
            iters,
            regrets,
            color=COND_COLORS[cid],
            linewidth=2,
            label=f"{COND_SHORT[cid]} (total: {regret_data[cid]['total_regret']:.1f})",
            alpha=0.9,
        )

    ax.set_title("Cumulative Regret Over Time", fontsize=14, fontweight="bold")
    ax.set_xlabel("Episode (task_order)")
    ax.set_ylabel("Cumulative Regret (sum of 1 - reward)")
    ax.legend(fontsize=10)

    plt.tight_layout()
    fig.savefig(fig_dir / "fig5_cumulative_regret.png")
    plt.close(fig)
    print("  Saved: fig5_cumulative_regret.png")


# ── Paper-Critical Figures (added for EMNLP submission) ────────────────────


def fig_multistep_decomposition(conn, fig_dir: Path):
    """Fig 8: Single-step vs multi-step token decomposition (paper Figure 3).

    KEY FIGURE: Demonstrates that all efficiency gains come from multi-step
    prevention, not retrieval improvement. Single-step bars should be equal
    height across conditions; multi-step bars drive the difference.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.5))

    for cid in range(1, 5):
        rows = conn.execute(
            "SELECT step_count, total_tokens FROM episodes "
            "WHERE condition_id = ? AND total_tokens IS NOT NULL "
            "AND step_count IS NOT NULL",
            (cid,),
        ).fetchall()
        steps = [r["step_count"] for r in rows]
        tokens = [r["total_tokens"] for r in rows]

        single = [t for s, t in zip(steps, tokens) if s == 1]
        multi = [t for s, t in zip(steps, tokens) if s > 1]
        n_total = len(tokens)

        # Store for stacked bar
        mean_single = float(np.mean(single)) if single else 0
        mean_multi = float(np.mean(multi)) if multi else 0
        pct_multi = 100 * len(multi) / n_total if n_total else 0

        x = cid

        # Panel 1: Stacked bar — mean tokens from single vs multi
        ax1.bar(
            x,
            mean_single,
            color=COND_COLORS[cid],
            alpha=0.5,
            edgecolor=DARK_FG,
            linewidth=0.5,
            label="Single-step" if cid == 1 else "",
        )
        ax1.bar(
            x,
            mean_multi,
            bottom=mean_single,
            color=COND_COLORS[cid],
            alpha=0.9,
            edgecolor=DARK_FG,
            linewidth=0.5,
            hatch="//",
            label="Multi-step" if cid == 1 else "",
        )

        # Panel 2: Multi-step rate
        ax2.bar(x, pct_multi, color=COND_COLORS[cid], alpha=0.8, edgecolor=DARK_FG, linewidth=0.5)
        ax2.text(
            x,
            pct_multi + 0.3,
            f"{pct_multi:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            color=DARK_FG,
        )

    ax1.set_xticks(range(1, 5))
    ax1.set_xticklabels([COND_SHORT[cid] for cid in range(1, 5)])
    ax1.set_title("Token Decomposition: Single vs Multi-Step", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Mean Tokens per Episode")
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))
    ax1.legend(fontsize=9, loc="upper right")

    ax2.set_xticks(range(1, 5))
    ax2.set_xticklabels([COND_SHORT[cid] for cid in range(1, 5)])
    ax2.set_title("Multi-Step Episode Rate", fontsize=11, fontweight="bold")
    ax2.set_ylabel("% Episodes with step_count > 1")
    ax2.set_ylim(0, max(1.0, ax2.get_ylim()[1] * 1.2))

    plt.tight_layout()
    fig.savefig(fig_dir / "fig8_multistep_decomposition.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig8_multistep_decomposition.png")


def fig_skill_retrieval_heatmap(conn, fig_dir: Path):
    """Fig A1: Top-20 skill retrieval frequency heatmap (skills x conditions).

    Shows whether the same skills dominate across all conditions (saturation)
    or different conditions retrieve different skills (differentiation).
    """
    # Get top-20 most retrieved skills overall
    rows = conn.execute("""
        SELECT rr.skill_id, s.title, COUNT(*) as cnt
        FROM retrieval_results rr
        JOIN skills s ON rr.skill_id = s.skill_id
        GROUP BY rr.skill_id
        ORDER BY cnt DESC
        LIMIT 20
    """).fetchall()

    if not rows:
        print("  [SKIP] fig_skill_retrieval_heatmap: no retrieval_results data")
        return

    skill_ids = [r["skill_id"] for r in rows]
    skill_names = [r["title"][:35] for r in rows]  # Truncate long names

    # Build frequency matrix: skills x conditions
    matrix = np.zeros((len(skill_ids), 4))
    for i, sid in enumerate(skill_ids):
        for j, cid in enumerate(range(1, 5)):
            cnt = conn.execute(
                "SELECT COUNT(*) FROM retrieval_results rr "
                "JOIN episodes e ON rr.episode_id = e.episode_id "
                "WHERE rr.skill_id = ? AND e.condition_id = ?",
                (sid, cid),
            ).fetchone()[0]
            matrix[i, j] = cnt

    fig, ax = plt.subplots(figsize=(10, 12))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(4))
    ax.set_xticklabels([COND_SHORT[cid] for cid in range(1, 5)])
    ax.set_yticks(range(len(skill_names)))
    ax.set_yticklabels(skill_names, fontsize=8)

    # Add text annotations
    max_val = matrix.max()
    for i in range(len(skill_ids)):
        for j in range(4):
            val = int(matrix[i, j])
            color = DARK_FG if (max_val == 0 or matrix[i, j] < max_val * 0.6) else DARK_BG
            ax.text(j, i, str(val), ha="center", va="center", fontsize=7, color=color)

    ax.set_title("Top-20 Skill Retrieval Frequency by Condition", fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Retrieval Count", shrink=0.8)
    plt.tight_layout()
    fig.savefig(fig_dir / "figA1_skill_retrieval_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: figA1_skill_retrieval_heatmap.png")


def fig_token_trend_over_time(conn, fig_dir: Path):
    """Fig A2: Token usage trends over time (rolling mean per condition).

    Shows whether conditions learn to be more efficient over episodes.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Adaptive window: use 20 or half the smallest condition size, whichever is smaller
    counts = [
        conn.execute(
            "SELECT COUNT(*) FROM episodes WHERE condition_id = ? AND total_tokens IS NOT NULL",
            (cid,),
        ).fetchone()[0]
        for cid in range(1, 5)
    ]
    min_count = max(min(c for c in counts if c > 0), 1) if any(c > 0 for c in counts) else 0
    if min_count < 4:
        print("  [SKIP] fig_token_trend_over_time: insufficient data for rolling mean")
        plt.close(fig)
        return
    window = min(20, min_count // 2)

    plotted_any = False
    for cid in range(1, 5):
        rows = conn.execute(
            "SELECT task_order, total_tokens FROM episodes "
            "WHERE condition_id = ? AND total_tokens IS NOT NULL "
            "ORDER BY task_order",
            (cid,),
        ).fetchall()
        if not rows:
            continue
        orders = [r["task_order"] for r in rows]
        tokens = [r["total_tokens"] for r in rows]

        if len(tokens) >= window:
            plotted_any = True
            rolling = np.convolve(tokens, np.ones(window) / window, mode="valid")
            x = orders[window - 1 :]
            ax.plot(
                x,
                rolling,
                color=COND_COLORS[cid],
                linewidth=2,
                label=f"{COND_SHORT[cid]} (rolling {window})",
                alpha=0.9,
            )

    if not plotted_any:
        plt.close(fig)
        print(f"  [SKIP] fig_token_trend_over_time: all conditions have < {window} episodes")
        return

    ax.set_title("Token Usage Over Time (Rolling Mean)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Task Order")
    ax.set_ylabel("Tokens per Episode")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))
    ax.legend(fontsize=10)

    plt.tight_layout()
    fig.savefig(fig_dir / "figA2_token_trend.png")
    plt.close(fig)
    print("  Saved: figA2_token_trend.png")


def fig_jaccard_vs_token_scatter(conn, fig_dir: Path):
    """Fig A4: Scatter plot of Jaccard similarity vs token difference per task.

    Tests whether tasks with lower skill overlap show larger token savings.
    """
    # Compare C1 vs C4 per task
    tasks = conn.execute("SELECT DISTINCT task_order FROM episodes ORDER BY task_order").fetchall()

    jaccards = []
    token_diffs = []

    for t in tasks:
        to = t["task_order"]
        # Get skill sets for C1 and C4
        c1_skills = set()
        c4_skills = set()
        c1_tokens = None
        c4_tokens = None

        for cid, skill_set in [(1, c1_skills), (4, c4_skills)]:
            ep = conn.execute(
                "SELECT episode_id, total_tokens FROM episodes "
                "WHERE condition_id = ? AND task_order = ?",
                (cid, to),
            ).fetchone()
            if not ep:
                continue
            if cid == 1:
                c1_tokens = ep["total_tokens"]
            else:
                c4_tokens = ep["total_tokens"]
            sids = conn.execute(
                "SELECT skill_id FROM retrieval_results WHERE episode_id = ?", (ep["episode_id"],)
            ).fetchall()
            for s in sids:
                skill_set.add(s["skill_id"])

        if c1_skills and c4_skills and c1_tokens is not None and c4_tokens is not None:
            intersection = len(c1_skills & c4_skills)
            union = len(c1_skills | c4_skills)
            j = intersection / union if union > 0 else 1.0
            jaccards.append(j)
            token_diffs.append(c1_tokens - c4_tokens)

    if not jaccards:
        print("  [SKIP] fig_jaccard_vs_token_scatter: insufficient data")
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(jaccards, token_diffs, alpha=0.5, color=COND_COLORS[4], s=30)

    ax.axhline(y=0, color=DARK_FG, linestyle="--", alpha=0.3)
    ax.set_title(
        "Jaccard Similarity vs Token Savings (C1 vs C4 per Task)", fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Jaccard Similarity (skill overlap)")
    ax.set_ylabel("Token Difference (C1 - C4)")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))

    # Add correlation if scipy available
    if HAS_SCIPY and len(jaccards) > 5:
        r, p = sp_stats.pearsonr(jaccards, token_diffs)
        ax.text(
            0.05,
            0.95,
            f"r = {r:.3f}, p = {p:.3f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor=DARK_ACCENT, alpha=0.8),
        )

    plt.tight_layout()
    fig.savefig(fig_dir / "figA4_jaccard_vs_tokens.png")
    plt.close(fig)
    print("  Saved: figA4_jaccard_vs_tokens.png")


def fig_rating_stability(conn, fig_dir: Path):
    """Fig A6: Feedback rating stability over time (evidence against reward hacking).

    Plots mean dimension ratings per quintile for C2/C3. No upward drift
    confirms input-assessment design avoids reward hacking (Pan et al., 2024).
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    dims = [
        ("rating_recency", "Recency"),
        ("rating_importance", "Importance"),
        ("rating_relevance", "Relevance"),
    ]

    _VALID_COLS = {"rating_recency", "rating_importance", "rating_relevance"}
    for ax, (col, label) in zip(axes, dims):
        assert col in _VALID_COLS, f"Invalid column: {col}"
        for cid in [2, 3]:
            rows = conn.execute(
                f"""
                SELECT e.task_order, f.{col}
                FROM feedback f
                JOIN episodes e ON f.episode_id = e.episode_id
                WHERE e.condition_id = ? AND f.{col} IS NOT NULL
                ORDER BY e.task_order
            """,
                (cid,),
            ).fetchall()

            if not rows:
                continue

            ratings = [r[col] for r in rows]

            # Quintile averages
            n = len(ratings)
            if n < 5:
                continue
            q_size = n // 5
            quintiles = []
            q_means = []
            for q in range(5):
                start = q * q_size
                end = start + q_size if q < 4 else n
                q_data = ratings[start:end]
                quintiles.append(f"Q{q + 1}")
                q_means.append(float(np.mean(q_data)))

            ax.plot(
                range(5),
                q_means,
                marker="o",
                linewidth=2,
                color=COND_COLORS[cid],
                label=COND_SHORT[cid],
                alpha=0.9,
            )

        ax.set_xticks(range(5))
        ax.set_xticklabels([f"Q{i + 1}" for i in range(5)])
        ax.set_title(f"{label} Rating", fontsize=12, fontweight="bold")
        ax.set_ylabel("Mean Rating")
        ax.legend(fontsize=9)
        ax.set_ylim(0, 5.5)

    fig.suptitle(
        "Rating Stability Over Time (Evidence Against Reward Hacking)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(fig_dir / "figA6_rating_stability.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: figA6_rating_stability.png")


def fig_bandit_posteriors(conn, fig_dir: Path):
    """Fig A7: Bandit posterior distributions (Beta PDFs for all 12 arms x 3 conditions).

    Shows the final posterior Beta distributions, allowing visual comparison
    of arm certainty and separation across conditions.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)
    x = np.linspace(0, 1, 200)

    # Color cycle for however many arms exist
    arm_cmap = plt.cm.tab20

    for idx, cid in enumerate([2, 3, 4]):
        ax = axes[idx]
        arms = conn.execute(
            "SELECT preset_id, alpha, beta, pulls FROM bandit_state "
            "WHERE condition_id = ? ORDER BY (alpha / (alpha + beta)) DESC",
            (cid,),
        ).fetchall()

        if not arms:
            ax.set_title(f"{COND_NAMES[cid]}\n(no data)", fontsize=12)
            continue

        n_arms = len(arms)
        arm_colors = [arm_cmap(i / max(n_arms, 1)) for i in range(n_arms)]

        for i, arm in enumerate(arms):
            a, b = arm["alpha"], arm["beta"]
            if HAS_SCIPY:
                pdf = sp_stats.beta.pdf(x, a, b)
                label = f"{arm['preset_id']} ({arm['pulls']}p, mu={a / (a + b):.3f})"
                ax.plot(x, pdf, color=arm_colors[i], linewidth=1.5, alpha=0.8, label=label)
            else:
                # Fallback: just mark the mean
                mean = a / (a + b)
                ax.axvline(
                    x=mean,
                    color=arm_colors[i],
                    linewidth=1.5,
                    alpha=0.8,
                    label=f"{arm['preset_id']} (mu={mean:.3f})",
                )

        ax.set_title(f"{COND_NAMES[cid]}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Reward")
        if idx == 0:
            ax.set_ylabel("Density")
        ax.legend(fontsize=6, loc="upper left", ncol=1)

    fig.suptitle(
        "Bandit Posterior Distributions (12 Arms)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(fig_dir / "figA7_bandit_posteriors.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: figA7_bandit_posteriors.png")


def fig_difficulty_comparison(conn, fig_dir: Path):
    """Fig A8: Per-difficulty reward comparison (grouped bars, 4 conditions x 3 difficulties)."""
    diff_data = compute_difficulty_breakdown(conn)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    diffs = ["easy", "medium", "hard"]
    x = np.arange(len(diffs))
    width = 0.2

    # Panel 1: Mean reward by difficulty
    for i, cid in enumerate(range(1, 5)):
        rewards = []
        for diff in diffs:
            r = diff_data.get(diff, {}).get(cid, {}).get("mean_reward")
            rewards.append(r if r is not None else 0)
        ax1.bar(
            x + i * width, rewards, width, label=COND_SHORT[cid], color=COND_COLORS[cid], alpha=0.8
        )
    ax1.set_xticks(x + 1.5 * width)
    ax1.set_xticklabels([d.capitalize() for d in diffs])
    ax1.set_title("Mean Reward by Difficulty", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Mean Reward")
    ax1.legend(fontsize=9)

    # Panel 2: Mean tokens by difficulty
    for i, cid in enumerate(range(1, 5)):
        tokens = []
        for diff in diffs:
            t = diff_data.get(diff, {}).get(cid, {}).get("mean_tokens")
            tokens.append(t if t is not None else 0)
        ax2.bar(
            x + i * width, tokens, width, label=COND_SHORT[cid], color=COND_COLORS[cid], alpha=0.8
        )
    ax2.set_xticks(x + 1.5 * width)
    ax2.set_xticklabels([d.capitalize() for d in diffs])
    ax2.set_title("Mean Tokens by Difficulty", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Tokens")
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))
    ax2.legend(fontsize=9)

    fig.suptitle("Per-Difficulty Analysis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(fig_dir / "figA8_difficulty_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: figA8_difficulty_comparison.png")


def fig_domain_heatmap(conn, fig_dir: Path):
    """Fig A9: Domain retrieval heatmap (10 domains x 4 conditions)."""
    domain_data = compute_domain_breakdown(conn)
    domains = domain_data["domains"]
    data = domain_data["data"]

    if not domains:
        print("  [SKIP] figA9_domain_heatmap.png (no domain data)")
        return

    matrix = np.zeros((len(domains), 4))
    for i, domain in enumerate(domains):
        for j, cid in enumerate(range(1, 5)):
            matrix[i, j] = data.get(domain, {}).get(cid, {}).get("top5_count", 0)

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.cm.YlOrRd
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")

    ax.set_xticks(range(4))
    ax.set_xticklabels([COND_SHORT[cid] for cid in range(1, 5)])
    ax.set_yticks(range(len(domains)))
    ax.set_yticklabels(domains, fontsize=9)

    for i in range(len(domains)):
        for j in range(4):
            val = int(matrix[i, j])
            max_val = matrix.max() if matrix.max() > 0 else 1
            color = DARK_FG if matrix[i, j] < max_val * 0.6 else DARK_BG
            ax.text(j, i, str(val), ha="center", va="center", fontsize=8, color=color)

    ax.set_title(
        "Skill Domain Retrieval Count (Top-5) by Condition", fontsize=13, fontweight="bold"
    )
    plt.colorbar(im, ax=ax, label="Retrieval Count", shrink=0.8)
    plt.tight_layout()
    fig.savefig(fig_dir / "figA9_domain_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: figA9_domain_heatmap.png")


# ── Output Formatting ───────────────────────────────────────────────────────


def print_metrics_table(metrics: list[dict]):
    """Print a formatted primary metrics table to stdout."""
    print("\n" + "=" * 100)
    print("PRIMARY METRICS TABLE")
    print("=" * 100)

    headers = [
        "Condition",
        "N",
        "Mean Rew",
        "Steps",
        "Tokens",
        "NDCG@5",
        "MRR",
        "Parse%",
        "Best Arm",
        "Post Mean",
    ]
    widths = [18, 5, 9, 7, 8, 7, 7, 7, 22, 9]
    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(header_line)
    print("-" * len(header_line))

    for m in metrics:
        vals = [
            m["condition"],
            str(m["episodes"]),
            f"{m['mean_reward']:.4f}" if m["mean_reward"] is not None else "N/A",
            f"{m['mean_steps']:.1f}" if m["mean_steps"] is not None else "N/A",
            f"{m['mean_tokens']:,.0f}" if m["mean_tokens"] is not None else "N/A",
            f"{m['ndcg5']:.4f}" if m["ndcg5"] is not None else "N/A",
            f"{m['mrr']:.4f}" if m["mrr"] is not None else "N/A",
            f"{m['parse_rate']:.0f}%" if m["cid"] > 1 else "---",
            m["best_arm"] or "---",
            f"{m['best_arm_posterior']:.4f}" if m["best_arm_posterior"] is not None else "---",
        ]
        print("  ".join(v.ljust(w) for v, w in zip(vals, widths)))

    print()


def print_statistical_results(stats: dict):
    """Print statistical test results."""
    print("=" * 100)
    print("STATISTICAL TESTS (each vs C1 Control)")
    print("=" * 100)

    if "error" in stats:
        print(f"  {stats['error']}")
        return

    for key in ["C2", "C3", "C4"]:
        s = stats.get(key)
        if not s:
            continue
        if "error" in s:
            print(f"\n  {key}: {s['error']}")
            continue

        print(f"\n  {key} vs C1:")
        ci = s["diff_ci_95"]
        print(f"    Mean diff:       {s['diff_mean']:+.4f}  [{ci[0]:+.4f}, {ci[1]:+.4f}]")
        print(f"    P(better):       {s['prob_treatment_better']:.4f}")
        print(f"    Mann-Whitney U:  {s['mann_whitney_U']:.1f}  p={s['mann_whitney_p']:.6f}")
        if "mann_whitney_p_bonferroni" in s:
            sig_b = "***" if s["significant_bonferroni_05"] else "n.s."
            print(f"    Bonferroni adj:  p={s['mann_whitney_p_bonferroni']:.6f}  {sig_b}")
        if "mann_whitney_p_bh" in s:
            sig_bh = "***" if s["significant_bh_05"] else "n.s."
            print(f"    BH adj:          p={s['mann_whitney_p_bh']:.6f}  {sig_bh}")
        if s["cohens_d"] is not None:
            print(f"    Cohen's d:       {s['cohens_d']:.4f}")
        else:
            print("    Cohen's d:       N/A")
        d = s.get("cohens_d")
        if d is not None:
            if abs(d) < 0.2:
                size = "negligible"
            elif abs(d) < 0.5:
                size = "small"
            elif abs(d) < 0.8:
                size = "medium"
            else:
                size = "large"
            print(f"    Effect size:     {size}")

    print()


def print_convergence(convergence: dict):
    """Print convergence analysis results."""
    print("=" * 100)
    print("CONVERGENCE ANALYSIS")
    print("=" * 100)

    for key in ["C2", "C3", "C4"]:
        c = convergence.get(key)
        if not c:
            continue
        stab = c.get("stabilized_at")
        total = c.get("total_episodes", 0)
        arm = c.get("best_arm", "?")
        window = c.get("stability_window", 20)

        if stab:
            print(f"  {key}: Stabilized at episode {stab}/{total} on '{arm}' (window={window})")
        else:
            print(
                f"  {key}: NOT stabilized after {total}"
                f" episodes (best arm: '{arm}',"
                f" window={window})"
            )

    print()


def print_theme_breakdown(breakdown: dict):
    """Print theme breakdown table."""
    print("=" * 100)
    print("THEME BREAKDOWN (Mean Reward)")
    print("=" * 100)

    themes = breakdown["themes"]
    matrix = breakdown["matrix"]
    variance = breakdown["theme_variance"]

    headers = ["Theme", "C1", "C2", "C3", "C4", "Variance"]
    widths = [28, 8, 8, 8, 8, 10]
    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(header_line)
    print("-" * len(header_line))

    # Sort by variance (highest first)
    sorted_themes = sorted(themes, key=lambda t: variance.get(t, 0), reverse=True)
    for theme in sorted_themes:
        vals = [
            (theme[:25] + "..." if len(theme) > 28 else theme),
        ]
        for cid in range(1, 5):
            v = matrix.get(theme, {}).get(cid)
            vals.append(f"{v:.4f}" if v is not None else "N/A")
        vals.append(f"{variance.get(theme, 0):.6f}")
        print("  ".join(v.ljust(w) for v, w in zip(vals, widths)))

    # Theme-specific presets
    presets = breakdown.get("theme_presets", {})
    if presets:
        print("\n  Most-used presets per theme:")
        for theme in sorted_themes:
            p = presets.get(theme, {})
            parts = []
            for cid in [2, 3, 4]:
                preset = p.get(cid, "?")
                parts.append(f"C{cid}={preset}")
            t_short = theme[:25] + "..." if len(theme) > 28 else theme
            print(f"    {t_short:28s}  {', '.join(parts)}")

    print()


def print_modality_comparison(modality: dict):
    """Print feedback modality comparison."""
    print("=" * 100)
    print("FEEDBACK MODALITY COMPARISON")
    print("=" * 100)

    c2c3 = modality.get("c2_vs_c3", {})
    print("\n  C2 (Likert only) vs C3 (Likert + Embedding):")
    print(f"    C2 mean reward:     {c2c3.get('c2_mean_reward', 'N/A')}")
    print(f"    C3 mean reward:     {c2c3.get('c3_mean_reward', 'N/A')}")
    if "mann_whitney_p" in c2c3:
        print(f"    Mann-Whitney p:     {c2c3['mann_whitney_p']:.6f}")
    if "cohens_d" in c2c3:
        print(f"    Cohen's d:          {c2c3['cohens_d']}")
    print(f"    C2 stabilized at:   {c2c3.get('c2_stabilized_at', 'N/A')}")
    print(f"    C3 stabilized at:   {c2c3.get('c3_stabilized_at', 'N/A')}")
    emb_diff = c2c3.get("embedding_differentiator", "N/A")
    c3_embs = c2c3.get("c3_embeddings_stored", 0)
    c2_embs = c2c3.get("c2_embeddings_stored", 0)
    print(f"    Embedding diff:     {emb_diff} (C3: {c3_embs} embs, C2: {c2_embs} embs)")

    svq = modality.get("structured_vs_qualitative", {})
    print("\n  Structured (C2+C3 pooled) vs Qualitative (C4):")
    s_mean = svq.get("structured_mean", "N/A")
    s_n = svq.get("structured_n", 0)
    q_mean = svq.get("qualitative_mean", "N/A")
    q_n = svq.get("qualitative_n", 0)
    print(f"    Structured mean:    {s_mean} (n={s_n})")
    print(f"    Qualitative mean:   {q_mean} (n={q_n})")
    if "mann_whitney_p" in svq:
        print(f"    Mann-Whitney p:     {svq['mann_whitney_p']:.6f}")
    if "cohens_d" in svq:
        print(f"    Cohen's d:          {svq['cohens_d']}")

    print("\n  Parse rates:")
    for cid in [2, 3, 4]:
        rate = modality.get(f"parse_rate_c{cid}", 0)
        print(f"    C{cid}: {rate:.1f}%")

    print()


# ── Public Export ─────────────────────────────────────────────────────────────


def export_public_results(conn, config_path=None):
    """Export curated result datasets to data/results/ for public release.

    Produces four files per PUBLIC_REPO_DESIGN.md:
      1. experiment_summary.json — aggregate per-condition metrics
      2. bandit_trajectories.csv — per-iteration bandit state (reconstructed)
      3. feedback_ratings.csv   — per-episode feedback (no raw prompts/responses)
      4. retrieval_metrics.csv  — per-episode retrieval quality (NDCG@5, MRR)

    No raw LLM prompts/responses are included (privacy).
    """
    import csv

    import yaml

    export_dir = ROOT / "data" / "results"
    export_dir.mkdir(parents=True, exist_ok=True)

    # Load experiment config for metadata
    config = {}
    cfg_path = config_path or ROOT / "configs" / "experiment.yaml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            config = yaml.safe_load(f) or {}

    total_episodes = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
    task_count = conn.execute("SELECT COUNT(DISTINCT task_id) FROM tasks").fetchone()[0]

    # ── 1. experiment_summary.json ────────────────────────────────────────────
    print("  [export 1/4] experiment_summary.json")

    exp_meta = {
        "tasks": task_count,
        "conditions": 4,
        "total_episodes": total_episodes,
        "seed": config.get("experiment", {}).get("seed", 42),
        "embedding_model": config.get("embeddings", {}).get("model", "Qwen/Qwen3-Embedding-0.6B"),
        "llm_model": config.get("llm", {}).get("model", "MiniMax-M2.5"),
    }

    # Per-condition metrics
    metrics = compute_primary_metrics(conn)
    conditions_block = {}
    for m in metrics:
        cid = m["cid"]
        conditions_block[str(cid)] = {
            "name": config.get("conditions", {}).get(cid, {}).get("name", COND_NAMES.get(cid, "")),
            "episodes": m["episodes"],
            "mean_tokens": m["mean_tokens"],
            "mean_steps": m["mean_steps"],
            "mean_reward": m["mean_reward"],
            "ndcg5": m["ndcg5"],
            "mrr": m["mrr"],
            "parse_rate": m["parse_rate"],
            "best_arm": m["best_arm"],
            "best_arm_posterior": m["best_arm_posterior"],
        }

    # Bandit final state (all arms for C2-C4)
    bandit_final = {}
    for cid in [2, 3, 4]:
        arms = conn.execute(
            "SELECT preset_id, alpha, beta, pulls, total_reward FROM bandit_state "
            "WHERE condition_id = ? ORDER BY (alpha / (alpha + beta)) DESC",
            (cid,),
        ).fetchall()
        bandit_final[str(cid)] = [
            {
                "preset_id": a["preset_id"],
                "alpha": round(a["alpha"], 4),
                "beta": round(a["beta"], 4),
                "mean": round(a["alpha"] / (a["alpha"] + a["beta"]), 4),
                "pulls": a["pulls"],
                "total_reward": round(a["total_reward"], 4),
            }
            for a in arms
        ]

    # Statistical tests
    stat_results = run_statistical_tests(conn)

    summary = {
        "experiment": exp_meta,
        "conditions": conditions_block,
        "bandit_final_state": bandit_final,
        "statistical_tests": stat_results,
    }

    summary_path = export_dir / "experiment_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"    Saved: {summary_path}")

    # ── 2. bandit_trajectories.csv ────────────────────────────────────────────
    #
    # Reconstruct per-iteration bandit state from episodes + feedback.
    # The bandit_state table only stores final cumulative values, so we replay
    # the sequential updates using the same Beta(α,β) update rule:
    #   α += reward, β += (1 - reward)
    # starting from the prior (α=1, β=1) for each arm.
    print("  [export 2/4] bandit_trajectories.csv")

    prior_alpha = config.get("bandit", {}).get("prior_alpha", 1.0)
    prior_beta = config.get("bandit", {}).get("prior_beta", 1.0)

    # Get all preset IDs
    preset_rows = conn.execute("SELECT preset_id FROM weight_presets ORDER BY preset_id").fetchall()
    all_presets = [r["preset_id"] for r in preset_rows]

    traj_path = export_dir / "bandit_trajectories.csv"
    with open(traj_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "task_order",
                "condition_id",
                "preset_id",
                "alpha",
                "beta",
                "mean",
                "pulls",
                "reward",
            ]
        )

        for cid in [2, 3, 4]:
            # Initialize arm state
            arm_state = {
                pid: {"alpha": prior_alpha, "beta": prior_beta, "pulls": 0} for pid in all_presets
            }

            # Get episodes with feedback, ordered by task_order
            rows = conn.execute(
                f"""
                SELECT e.task_order, e.preset_id,
                       {_reward_sql()} as reward
                FROM episodes e
                LEFT JOIN feedback f ON f.episode_id = e.episode_id
                WHERE e.condition_id = ?
                ORDER BY e.task_order
            """,
                (cid,),
            ).fetchall()

            for row in rows:
                task_order = row["task_order"]
                preset_id = row["preset_id"]
                reward = row["reward"]

                if preset_id is None:
                    continue

                # Update the arm that was pulled
                if reward is not None:
                    arm_state[preset_id]["alpha"] += reward
                    arm_state[preset_id]["beta"] += 1.0 - reward
                    arm_state[preset_id]["pulls"] += 1

                # Write a row for the pulled arm at this task_order
                st = arm_state[preset_id]
                mean = st["alpha"] / (st["alpha"] + st["beta"])
                writer.writerow(
                    [
                        task_order,
                        cid,
                        preset_id,
                        round(st["alpha"], 4),
                        round(st["beta"], 4),
                        round(mean, 4),
                        st["pulls"],
                        round(reward, 6) if reward is not None else "",
                    ]
                )

    print(f"    Saved: {traj_path}")

    # ── 3. feedback_ratings.csv ───────────────────────────────────────────────
    print("  [export 3/4] feedback_ratings.csv")

    fb_path = export_dir / "feedback_ratings.csv"
    with open(fb_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "episode_id",
                "condition_id",
                "task_id",
                "task_order",
                "theme",
                "difficulty",
                "preset_id",
                "rating_recency",
                "rating_importance",
                "rating_relevance",
                "inferred_recency",
                "inferred_importance",
                "inferred_relevance",
                "composite_reward",
                "total_tokens",
                "input_tokens",
                "output_tokens",
                "step_count",
                "success",
            ]
        )

        rows = conn.execute(f"""
            SELECT e.episode_id, e.condition_id, e.task_id, e.task_order,
                   t.theme, t.difficulty, e.preset_id,
                   f.rating_recency, f.rating_importance, f.rating_relevance,
                   f.inferred_recency, f.inferred_importance, f.inferred_relevance,
                   {_reward_sql()} as composite_reward,
                   e.total_tokens, e.input_tokens, e.output_tokens,
                   e.step_count, e.success
            FROM episodes e
            JOIN tasks t ON e.task_id = t.task_id
            LEFT JOIN feedback f ON f.episode_id = e.episode_id
            ORDER BY e.condition_id, e.task_order
        """).fetchall()

        for row in rows:
            writer.writerow(
                [
                    row["episode_id"],
                    row["condition_id"],
                    row["task_id"],
                    row["task_order"],
                    row["theme"],
                    row["difficulty"],
                    row["preset_id"],
                    row["rating_recency"],
                    row["rating_importance"],
                    row["rating_relevance"],
                    _fmt(row["inferred_recency"]),
                    _fmt(row["inferred_importance"]),
                    _fmt(row["inferred_relevance"]),
                    round(row["composite_reward"], 6)
                    if row["composite_reward"] is not None
                    else "",
                    row["total_tokens"],
                    row["input_tokens"],
                    row["output_tokens"],
                    row["step_count"],
                    row["success"],
                ]
            )

    print(f"    Saved: {fb_path}")

    # ── 4. retrieval_metrics.csv ──────────────────────────────────────────────
    print("  [export 4/4] retrieval_metrics.csv")

    ret_path = export_dir / "retrieval_metrics.csv"
    with open(ret_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "episode_id",
                "condition_id",
                "task_id",
                "ndcg5",
                "mrr",
                "top1_skill",
                "top1_is_gt",
                "num_gt_in_top5",
                "preset_id",
            ]
        )

        # Get all episodes that have retrieval results
        episodes = conn.execute("""
            SELECT e.episode_id, e.condition_id, e.task_id, e.preset_id,
                   t.ground_truth_skills
            FROM episodes e
            JOIN tasks t ON e.task_id = t.task_id
            ORDER BY e.condition_id, e.task_order
        """).fetchall()

        for ep in episodes:
            ep_id = ep["episode_id"]

            # Get top-5 retrieved skills for this episode
            retrieved = conn.execute(
                "SELECT skill_id, is_ground_truth FROM retrieval_results "
                "WHERE episode_id = ? ORDER BY rank LIMIT 5",
                (ep_id,),
            ).fetchall()

            if not retrieved:
                continue

            # Parse ground truth
            gt_set = set()
            try:
                gt_skills = json.loads(ep["ground_truth_skills"] or "[]")
                gt_set = set(gt_skills)
            except (json.JSONDecodeError, TypeError):
                gt_skills = []

            # NDCG@5
            k = min(5, len(retrieved))
            dcg, idcg = 0.0, 0.0
            for i in range(k):
                rel = 1.0 if retrieved[i]["skill_id"] in gt_set else 0.0
                dcg += rel / math.log2(i + 2)
            for i in range(min(k, len(gt_skills))):
                idcg += 1.0 / math.log2(i + 2)
            ndcg5 = round(dcg / idcg, 4) if idcg > 0 else 0.0

            # MRR
            mrr = 0.0
            for i, r in enumerate(retrieved):
                if r["skill_id"] in gt_set:
                    mrr = round(1.0 / (i + 1), 4)
                    break

            # Top-1 info
            top1_skill = retrieved[0]["skill_id"]
            top1_is_gt = 1 if top1_skill in gt_set else 0

            # Count GT hits in top-5
            num_gt_in_top5 = sum(1 for r in retrieved if r["skill_id"] in gt_set)

            writer.writerow(
                [
                    ep_id,
                    ep["condition_id"],
                    ep["task_id"],
                    ndcg5,
                    mrr,
                    top1_skill,
                    top1_is_gt,
                    num_gt_in_top5,
                    ep["preset_id"],
                ]
            )

    print(f"    Saved: {ret_path}")

    print(f"\n  Export complete: {export_dir}")
    print(
        "  Files: experiment_summary.json, bandit_trajectories.csv, "
        "feedback_ratings.csv, retrieval_metrics.csv"
    )


def _fmt(val):
    """Format a float for CSV output, returning empty string for None."""
    if val is None:
        return ""
    return round(val, 6)


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Phase 4 — Comprehensive experiment analysis (v3)")
    parser.add_argument(
        "--db",
        default="experiment_v3.db",
        help="Path to experiment database (default: experiment_v3.db)",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export curated result datasets to data/results/ for public release",
    )
    args = parser.parse_args()

    setup_dark_theme()

    conn = get_db(args.db)

    # Ensure output directories exist
    results_dir = ROOT / "results"
    fig_dir = results_dir / "figures"
    latex_dir = results_dir / "latex"
    results_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)
    latex_dir.mkdir(exist_ok=True)

    total_episodes = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
    print(f"\nAnalyzing {args.db}: {total_episodes} episodes found\n")

    # Load external ground truth
    gt_lookup = load_external_ground_truth()
    if gt_lookup:
        print(
            f"  Loaded ground_truth_v3.json: {len(gt_lookup)} tasks, "
            f"{sum(len(v['relevant']) for v in gt_lookup.values())} assignments\n"
        )

    import csv

    # ── 1. Primary Metrics ──
    print("[1/15] Computing primary metrics...")
    metrics = compute_primary_metrics(conn)
    print_metrics_table(metrics)

    csv_path = results_dir / "primary_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)
    print(f"  Saved: {csv_path.name}")

    emit_latex_primary_table(metrics, latex_dir / "tab_primary_metrics.tex")
    print("  Saved: latex/tab_primary_metrics.tex")

    # ── 2. Statistical Tests ──
    print("\n[2/15] Running statistical tests...")
    stat_results = run_statistical_tests(conn)
    print_statistical_results(stat_results)

    if "error" not in stat_results:
        emit_latex_statistical_table(stat_results, latex_dir / "tab_stat_tests.tex")
        print("  Saved: latex/tab_stat_tests.tex")

    # ── 2b. Token Consumption Statistical Tests ──
    print("\n[2b/15] Running token consumption statistical tests...")
    token_stat_results = run_token_statistical_tests(conn)
    print_token_statistical_results(token_stat_results)

    token_stats_path = results_dir / "statistical_tests.json"
    with open(token_stats_path, "w") as f:
        json.dump(
            {
                "description": "Token consumption statistical tests across experiment conditions",
                "reward_tests": stat_results,
                "token_tests": token_stat_results,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"  Saved: {token_stats_path.name}")

    # ── 3. Convergence Analysis ──
    print("[3/15] Analyzing convergence...")
    convergence = compute_convergence(conn)
    print_convergence(convergence)

    # ── 4. Theme Breakdown ──
    print("[4/15] Computing theme breakdown...")
    theme_breakdown = compute_theme_breakdown(conn)
    print_theme_breakdown(theme_breakdown)

    themes = theme_breakdown["themes"]
    matrix = theme_breakdown["matrix"]
    theme_csv = results_dir / "theme_breakdown.csv"
    with open(theme_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Theme", "C1_reward", "C2_reward", "C3_reward", "C4_reward", "variance"])
        for theme in themes:
            row = [theme]
            for cid in range(1, 5):
                row.append(matrix.get(theme, {}).get(cid, ""))
            row.append(theme_breakdown["theme_variance"].get(theme, ""))
            writer.writerow(row)
    print(f"  Saved: {theme_csv.name}")

    # ── 5. Feedback Modality Comparison ──
    print("\n[5/15] Comparing feedback modalities...")
    modality = compute_modality_comparison(conn)
    print_modality_comparison(modality)

    # ── 6. Ground Truth Hit Rate (DB-based) ──
    print("[6/15] Computing ground truth hit rates (DB-based)...")
    ground_truth = compute_ground_truth_hit_rate(conn)
    print_ground_truth(ground_truth)

    # ── 7. Ground Truth — Graded NDCG/MRR (external file) ──
    if gt_lookup:
        print("[7/15] Computing graded NDCG/MRR (ground_truth_v3.json)...")
        gt_external = compute_ground_truth_external(conn, gt_lookup)
        print_ground_truth_external(gt_external)
        emit_latex_graded_gt_table(gt_external, latex_dir / "tab_graded_gt.tex")
        print("  Saved: latex/tab_graded_gt.tex")
    else:
        print("[7/15] Skipped — ground_truth_v3.json not found")
        gt_external = {}

    # ── 8. Jaccard Similarity ──
    print("[8/15] Analyzing skill set overlap (Jaccard)...")
    jaccard = compute_jaccard_trajectories(conn)
    print_jaccard(jaccard)
    emit_latex_jaccard_table(jaccard, latex_dir / "tab_jaccard.tex")
    print("  Saved: latex/tab_jaccard.tex")

    # ── 9. Token Breakdown ──
    print("[9/15] Analyzing token input/output breakdown...")
    token_breakdown = compute_token_breakdown(conn)
    print_token_breakdown(token_breakdown)

    # ── 10. Anchor Spread Validation ──
    print("[10/15] Validating C4 anchor spread...")
    anchor = compute_anchor_validation(conn)
    print_anchor_validation(anchor)

    # ── 11. Per-Difficulty Breakdown ──
    print("[11/15] Computing per-difficulty breakdown (50E/100M/150H)...")
    diff_data = compute_difficulty_breakdown(conn)
    print_difficulty_breakdown(diff_data)
    emit_latex_difficulty_table(diff_data, latex_dir / "tab_difficulty.tex")
    print("  Saved: latex/tab_difficulty.tex")

    # ── 12. Per-Domain Analysis ──
    print("[12/15] Computing per-domain analysis (10 domains)...")
    domain_data = compute_domain_breakdown(conn)
    print_domain_breakdown(domain_data)

    # ── 13. Figures ──
    print("[13/15] Generating figures (15 total)...")
    figure_funcs = [
        fig_reward_boxplot,
        fig_convergence_curves,
        fig_theme_heatmap,
        fig_step_efficiency,
        fig_cumulative_regret,
        fig_ground_truth_hit_rate,
        fig_jaccard_trajectory,
        fig_multistep_decomposition,
        fig_skill_retrieval_heatmap,
        fig_token_trend_over_time,
        fig_jaccard_vs_token_scatter,
        fig_rating_stability,
        fig_bandit_posteriors,
        fig_difficulty_comparison,
        fig_domain_heatmap,
    ]
    for func in figure_funcs:
        try:
            func(conn, fig_dir)
        except Exception as e:
            print(f"  [ERROR] {func.__name__} failed: {e}")
            plt.close("all")

    # ── 14. Save comprehensive JSON summary ──
    print("\n[14/15] Writing JSON summary...")
    # Strip non-serializable data from convergence (keep summary only)
    convergence_summary = {}
    for key in ["C2", "C3", "C4"]:
        c = convergence.get(key, {})
        convergence_summary[key] = {
            "stabilized_at": c.get("stabilized_at"),
            "total_episodes": c.get("total_episodes"),
            "best_arm": c.get("best_arm"),
            "stability_window": c.get("stability_window"),
        }

    # Regret totals
    regret_data = compute_cumulative_regret_curves(conn)
    regret_summary = {COND_SHORT[cid]: regret_data[cid]["total_regret"] for cid in range(1, 5)}

    # Ground truth summary (strip by_theme/by_difficulty for compactness)
    gt_summary = {}
    for cid in range(1, 5):
        g = ground_truth.get(cid, {})
        gt_summary[COND_SHORT[cid]] = {
            "hit_rate": g.get("hit_rate"),
            "mean_gt_in_top5": g.get("mean_gt_in_top5"),
            "n": g.get("n", 0),
        }

    # Graded GT summary
    gt_ext_summary = {}
    for cid in range(1, 5):
        g = gt_external.get(cid, {})
        gt_ext_summary[COND_SHORT[cid]] = {
            "ndcg5_graded": g.get("ndcg5"),
            "mrr": g.get("mrr"),
            "count": g.get("count", 0),
        }

    # Jaccard summary (overall means only)
    jaccard_summary = {}
    for key, data in jaccard.get("pairs", {}).items():
        jaccard_summary[key] = {
            "overall_mean": data.get("overall_mean"),
            "perfect_overlap_pct": data.get("perfect_overlap_pct"),
        }

    # Token summary
    token_summary = {}
    for cid in range(1, 5):
        t = token_breakdown.get(cid, {})
        token_summary[COND_SHORT[cid]] = {
            "mean_input": t.get("mean_input"),
            "mean_output": t.get("mean_output"),
            "mean_total": t.get("mean_total"),
            "input_pct": t.get("input_pct"),
            "sum_total": t.get("sum_total"),
        }

    # Anchor summary
    anchor_summary = anchor.get("dimensions", {})

    summary = {
        "database": args.db,
        "total_episodes": total_episodes,
        "schema_version": "v3",
        "presets": 12,
        "tasks": 300,
        "difficulty_distribution": {"easy": 50, "medium": 100, "hard": 150},
        "primary_metrics": metrics,
        "statistical_tests": stat_results,
        "token_statistical_tests": token_stat_results,
        "convergence": convergence_summary,
        "theme_breakdown": {
            "matrix": theme_breakdown["matrix"],
            "theme_variance": theme_breakdown["theme_variance"],
            "theme_presets": theme_breakdown.get("theme_presets", {}),
        },
        "difficulty_breakdown": diff_data,
        "domain_breakdown": {
            d: {
                "skill_count": domain_data["data"][d]["skill_count"],
                **{
                    f"C{cid}_top5": domain_data["data"][d].get(cid, {}).get("top5_count", 0)
                    for cid in range(1, 5)
                },
            }
            for d in domain_data["domains"]
            if d in domain_data["data"]
        },
        "modality_comparison": {
            k: v
            for k, v in modality.items()
            if not k.startswith("parse_rate_trend")  # exclude large trend arrays
        },
        "ground_truth": gt_summary,
        "ground_truth_graded": gt_ext_summary,
        "jaccard_similarity": jaccard_summary,
        "token_breakdown": token_summary,
        "anchor_validation": anchor_summary,
        "cumulative_regret": regret_summary,
    }

    summary_path = results_dir / "analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved: {summary_path.name}")

    # ── 15. Per-condition reward CSVs ──
    print("[15/15] Writing per-condition reward CSVs...")
    rewards_ordered = get_rewards_per_condition_ordered(conn)
    for cid in range(1, 5):
        cond_csv = results_dir / f"rewards_{COND_SHORT[cid].lower()}.csv"
        with open(cond_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["task_order", "reward"])
            for entry in rewards_ordered[cid]:
                writer.writerow([entry["task_order"], round(entry["reward"], 6)])
        print(f"  Saved: {cond_csv.name}")

    # ── Optional: Public Export ──
    if args.export:
        print("\n[EXPORT] Writing public release datasets to data/results/...")
        export_public_results(conn)

    conn.close()

    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE")
    print(f"  Results:  {results_dir}")
    print(f"  Figures:  {fig_dir}")
    print(f"  LaTeX:    {latex_dir}")
    print(f"  Summary:  {summary_path}")
    if args.export:
        print(f"  Export:   {ROOT / 'data' / 'results'}")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
