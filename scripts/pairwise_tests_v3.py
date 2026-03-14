"""
Pairwise statistical tests for v3 experiment data.

Tests all 6 pairwise condition comparisons on 4 metrics:
  - total_tokens, output_tokens, step_count, duration_ms

For each pair:
  - Welch's t-test (unequal variance)
  - Mann-Whitney U test (non-parametric)
  - Cohen's d effect size
  - 95% CI of mean difference
  - Holm-Bonferroni correction across all 24 tests per test type
"""

import sqlite3
from itertools import combinations

import numpy as np
from scipy import stats

DB_PATH = __import__("pathlib").Path(__file__).resolve().parent.parent / "experiment_v3.db"

METRICS = ["total_tokens", "output_tokens", "step_count", "duration_ms"]
CONDITIONS = [1, 2, 3, 4]
CONDITION_LABELS = {
    1: "C1 (No Retrieval)",
    2: "C2 (Static Weights)",
    3: "C3 (Full System)",
    4: "C4 (Bandit Only)",
}
ALPHA = 0.05


def load_data(db_path: str) -> dict[int, dict[str, np.ndarray]]:
    """Load episode data grouped by condition."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    data: dict[int, dict[str, list]] = {c: {m: [] for m in METRICS} for c in CONDITIONS}
    cur.execute(f"SELECT condition_id, {', '.join(METRICS)} FROM episodes")
    for row in cur.fetchall():
        cid = row[0]
        for i, m in enumerate(METRICS):
            val = row[i + 1]
            if val is not None:
                data[cid][m].append(val)
    conn.close()
    # Convert to numpy arrays
    return {c: {m: np.array(vals) for m, vals in metrics.items()} for c, metrics in data.items()}


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d with pooled SD (using Hedges' correction for unequal n)."""
    na, nb = len(a), len(b)
    var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_sd = np.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2))
    if pooled_sd == 0:
        return 0.0
    d = (np.mean(a) - np.mean(b)) / pooled_sd
    # Hedges' g correction for small-sample bias
    correction = 1 - 3 / (4 * (na + nb) - 9)
    return d * correction


def ci_mean_diff(a: np.ndarray, b: np.ndarray, confidence: float = 0.95):
    """95% CI for difference in means using Welch-Satterthwaite df."""
    diff = np.mean(a) - np.mean(b)
    se = np.sqrt(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b))
    # Welch-Satterthwaite degrees of freedom
    va, vb = np.var(a, ddof=1) / len(a), np.var(b, ddof=1) / len(b)
    df_num = (va + vb) ** 2
    df_den = va**2 / (len(a) - 1) + vb**2 / (len(b) - 1)
    df = df_num / df_den if df_den > 0 else len(a) + len(b) - 2
    t_crit = stats.t.ppf((1 + confidence) / 2, df)
    return diff - t_crit * se, diff + t_crit * se


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """
    Holm-Bonferroni correction. Returns list of booleans (significant or not).
    """
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    significant = [False] * m
    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted_alpha = alpha / (m - rank)
        if p <= adjusted_alpha:
            significant[orig_idx] = True
        else:
            # Once one fails, all remaining are non-significant
            break
    return significant


def effect_size_label(d: float) -> str:
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


def print_descriptives(data: dict):
    """Print descriptive statistics for each condition and metric."""
    print("=" * 100)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 100)
    for metric in METRICS:
        print(f"\n--- {metric} ---")
        print(
            f"{'Condition':<25} {'N':>5} {'Mean':>12} {'SD':>12} "
            f"{'Median':>12} {'Min':>10} {'Max':>10} {'Skew':>8} {'Kurt':>8}"
        )
        print("-" * 110)
        for c in CONDITIONS:
            arr = data[c][metric]
            print(
                f"{CONDITION_LABELS[c]:<25} {len(arr):>5} {np.mean(arr):>12.1f} "
                f"{np.std(arr, ddof=1):>12.1f} {np.median(arr):>12.1f} "
                f"{np.min(arr):>10.0f} {np.max(arr):>10.0f} "
                f"{stats.skew(arr):>8.2f} {stats.kurtosis(arr):>8.2f}"
            )


def print_normality(data: dict):
    """Shapiro-Wilk and D'Agostino-Pearson normality tests."""
    print("\n" + "=" * 100)
    print("NORMALITY TESTS (justification for non-parametric backup)")
    print("=" * 100)
    for metric in METRICS:
        print(f"\n--- {metric} ---")
        print(f"{'Condition':<25} {'Shapiro-Wilk W':>15} {'p-value':>12} {'Normal?':>10}")
        print("-" * 65)
        for c in CONDITIONS:
            arr = data[c][metric]
            w, p = stats.shapiro(arr)
            normal = "Yes" if p > 0.05 else "No"
            print(f"{CONDITION_LABELS[c]:<25} {w:>15.6f} {p:>12.2e} {normal:>10}")


def run_tests(data: dict):
    """Run all pairwise tests and print results."""
    pairs = list(combinations(CONDITIONS, 2))

    # Collect all p-values for Holm-Bonferroni correction
    all_welch_p: list[tuple[int, int, str, float]] = []  # (c1, c2, metric, p)
    all_mw_p: list[tuple[int, int, str, float]] = []

    # Store results for final summary
    results = {}

    for metric in METRICS:
        for c1, c2 in pairs:
            a = data[c1][metric]
            b = data[c2][metric]

            # Welch's t-test
            t_stat, t_p = stats.ttest_ind(a, b, equal_var=False)

            # Mann-Whitney U
            u_stat, u_p = stats.mannwhitneyu(a, b, alternative="two-sided")

            # Cohen's d (Hedges' g)
            d = cohens_d(a, b)

            # 95% CI of mean difference
            ci_lo, ci_hi = ci_mean_diff(a, b)

            # Mean difference
            mean_diff = np.mean(a) - np.mean(b)

            key = (c1, c2, metric)
            results[key] = {
                "mean_a": np.mean(a),
                "mean_b": np.mean(b),
                "mean_diff": mean_diff,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "t_stat": t_stat,
                "t_p": t_p,
                "u_stat": u_stat,
                "u_p": u_p,
                "d": d,
                "d_label": effect_size_label(d),
                "n_a": len(a),
                "n_b": len(b),
            }

            all_welch_p.append((c1, c2, metric, t_p))
            all_mw_p.append((c1, c2, metric, u_p))

    # Holm-Bonferroni correction
    welch_pvals = [x[3] for x in all_welch_p]
    mw_pvals = [x[3] for x in all_mw_p]
    welch_sig = holm_bonferroni(welch_pvals, ALPHA)
    mw_sig = holm_bonferroni(mw_pvals, ALPHA)

    for i, (c1, c2, metric, _) in enumerate(all_welch_p):
        key = (c1, c2, metric)
        results[key]["welch_holm_sig"] = welch_sig[i]
    for i, (c1, c2, metric, _) in enumerate(all_mw_p):
        key = (c1, c2, metric)
        results[key]["mw_holm_sig"] = mw_sig[i]

    # Print per-metric detailed results
    for metric in METRICS:
        print("\n" + "=" * 100)
        print(f"PAIRWISE TESTS: {metric}")
        print("=" * 100)

        header = (
            f"{'Pair':<18} {'Mean Diff':>11} {'95% CI':>24} "
            f"{'t':>8} {'p(Welch)':>12} {'Holm':>5} "
            f"{'U':>12} {'p(MW)':>12} {'Holm':>5} "
            f"{'d':>7} {'Size':>10}"
        )
        print(header)
        print("-" * len(header))

        for c1, c2 in pairs:
            key = (c1, c2, metric)
            r = results[key]
            ci_str = f"[{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}]"
            w_sig = "Y" if r["welch_holm_sig"] else "N"
            m_sig = "Y" if r["mw_holm_sig"] else "N"

            print(
                f"C{c1} vs C{c2}        "
                f"{r['mean_diff']:>+11.1f} {ci_str:>24} "
                f"{r['t_stat']:>8.3f} {r['t_p']:>12.2e} {w_sig:>5} "
                f"{r['u_stat']:>12.0f} {r['u_p']:>12.2e} {m_sig:>5} "
                f"{r['d']:>+7.3f} {r['d_label']:>10}"
            )

    # Grand summary of significant results
    print("\n" + "=" * 100)
    print("SUMMARY: SIGNIFICANT RESULTS AFTER HOLM-BONFERRONI CORRECTION")
    print("=" * 100)

    print("\n--- Welch's t-test (Holm-corrected, alpha=0.05) ---")
    any_sig = False
    for metric in METRICS:
        for c1, c2 in pairs:
            key = (c1, c2, metric)
            r = results[key]
            if r["welch_holm_sig"]:
                any_sig = True
                print(
                    f"  {metric:<16} C{c1} vs C{c2}: "
                    f"diff={r['mean_diff']:+.1f}, "
                    f"95%CI=[{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}], "
                    f"t={r['t_stat']:.3f}, p={r['t_p']:.2e}, "
                    f"d={r['d']:+.3f} ({r['d_label']})"
                )
    if not any_sig:
        print("  None")

    print("\n--- Mann-Whitney U (Holm-corrected, alpha=0.05) ---")
    any_sig = False
    for metric in METRICS:
        for c1, c2 in pairs:
            key = (c1, c2, metric)
            r = results[key]
            if r["mw_holm_sig"]:
                any_sig = True
                print(
                    f"  {metric:<16} C{c1} vs C{c2}: "
                    f"diff={r['mean_diff']:+.1f}, "
                    f"U={r['u_stat']:.0f}, p={r['u_p']:.2e}, "
                    f"d={r['d']:+.3f} ({r['d_label']})"
                )
    if not any_sig:
        print("  None")

    # Effect size summary
    print("\n" + "=" * 100)
    print("EFFECT SIZE MATRIX (Cohen's d / Hedges' g)")
    print("=" * 100)
    for metric in METRICS:
        print(f"\n--- {metric} ---")
        print(f"{'':>18}", end="")
        for c2 in CONDITIONS[1:]:
            print(f"{'C' + str(c2):>12}", end="")
        print()
        for c1 in CONDITIONS[:-1]:
            print(f"C{c1:<17}", end="")
            for c2 in CONDITIONS[1:]:
                if c2 <= c1:
                    print(f"{'':>12}", end="")
                else:
                    key = (c1, c2, metric)
                    r = results[key]
                    print(f"{r['d']:>+12.3f}", end="")
            print()

    # Agreement check between parametric and non-parametric
    print("\n" + "=" * 100)
    print("AGREEMENT CHECK: Welch vs Mann-Whitney significance")
    print("=" * 100)
    agree = 0
    disagree = 0
    total = 0
    for metric in METRICS:
        for c1, c2 in pairs:
            key = (c1, c2, metric)
            r = results[key]
            total += 1
            w = r["welch_holm_sig"]
            m = r["mw_holm_sig"]
            if w == m:
                agree += 1
            else:
                disagree += 1
                print(
                    f"  DISAGREE: {metric} C{c1} vs C{c2} — "
                    f"Welch={'sig' if w else 'ns'}, MW={'sig' if m else 'ns'} "
                    f"(Welch p={r['t_p']:.2e}, MW p={r['u_p']:.2e})"
                )
    print(f"\n  Agreement: {agree}/{total} ({100 * agree / total:.0f}%)")
    if disagree == 0:
        print("  All tests agree — parametric and non-parametric conclusions match.")

    return results


def main():
    print("Loading data from:", DB_PATH)
    data = load_data(str(DB_PATH))

    for c in CONDITIONS:
        n = len(data[c][METRICS[0]])
        print(f"  {CONDITION_LABELS[c]}: n={n}")

    print_descriptives(data)
    print_normality(data)
    results = run_tests(data)

    # Count total significant
    n_welch = sum(1 for r in results.values() if r["welch_holm_sig"])
    n_mw = sum(1 for r in results.values() if r["mw_holm_sig"])
    print(f"\nTotal significant (Welch, Holm-corrected): {n_welch}/24")
    print(f"Total significant (MW, Holm-corrected): {n_mw}/24")


if __name__ == "__main__":
    main()
