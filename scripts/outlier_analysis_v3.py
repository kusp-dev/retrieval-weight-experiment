"""
Outlier analysis for v3 experiment data.
Analyzes total_tokens at multiple SD cutoffs and winsorization levels.
"""

import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

DB_PATH = Path(__file__).resolve().parent.parent / "experiment_v3.db"


def load_data():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT e.episode_id, e.condition_id, e.task_id, e.preset_id,
               e.step_count, e.total_tokens,
               t.title as task_title, t.theme as task_theme
        FROM episodes e
        JOIN tasks t ON e.task_id = t.task_id
        WHERE e.total_tokens IS NOT NULL
        ORDER BY e.condition_id, e.episode_id
    """)
    rows = cursor.fetchall()
    conn.close()
    return rows


def section_header(title):
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


def part1_identify_outliers(rows):
    """Identify outliers at +1, +2, +3 SD cutoffs for total_tokens."""
    section_header("1. OUTLIER IDENTIFICATION AT MULTIPLE SD CUTOFFS")

    all_tokens = np.array([r["total_tokens"] for r in rows])
    overall_mean = np.mean(all_tokens)
    overall_sd = np.std(all_tokens, ddof=1)

    print(f"Overall statistics (N={len(all_tokens)}):")
    print(f"  Mean:   {overall_mean:,.1f}")
    print(f"  SD:     {overall_sd:,.1f}")
    print(f"  Median: {np.median(all_tokens):,.1f}")
    print(f"  Min:    {np.min(all_tokens):,}")
    print(f"  Max:    {np.max(all_tokens):,}")
    print(f"  Skew:   {stats.skew(all_tokens):.3f}")
    print(f"  Kurt:   {stats.kurtosis(all_tokens):.3f}")

    # Per-condition stats
    conditions = sorted(set(r["condition_id"] for r in rows))
    cond_data = {}
    for c in conditions:
        tokens = np.array([r["total_tokens"] for r in rows if r["condition_id"] == c])
        cond_data[c] = tokens
        print(
            f"\n  C{c} (n={len(tokens)}): mean={np.mean(tokens):,.1f}, "
            f"sd={np.std(tokens, ddof=1):,.1f}, "
            f"median={np.median(tokens):,.1f}, "
            f"mean/median={np.mean(tokens) / np.median(tokens):.2f}"
        )

    # Global SD cutoff analysis
    print("\n--- Global SD cutoff analysis ---")
    print(f"{'Cutoff':<12} {'Threshold':>12} {'N outliers':>12} {'% of total':>12}")
    print("-" * 52)

    for k in [1, 2, 3]:
        threshold = overall_mean + k * overall_sd
        n_outliers = np.sum(all_tokens > threshold)
        pct = 100 * n_outliers / len(all_tokens)
        print(f"+{k} SD       {threshold:>12,.0f} {n_outliers:>12} {pct:>11.1f}%")

    # Breakdown by condition
    print("\n--- Outlier breakdown by condition (global SD thresholds) ---")
    for k in [1, 2, 3]:
        threshold = overall_mean + k * overall_sd
        print(f"\n  +{k} SD (threshold = {threshold:,.0f}):")
        for c in conditions:
            n_out = np.sum(cond_data[c] > threshold)
            pct = 100 * n_out / len(cond_data[c])
            print(f"    C{c}: {n_out:>4} outliers ({pct:>5.1f}% of C{c})")

    # Per-condition SD cutoff (as paper describes)
    print("\n--- Outlier breakdown (per-condition SD thresholds) ---")
    for k in [1, 2, 3]:
        print(f"\n  +{k} SD (condition-specific thresholds):")
        for c in conditions:
            c_mean = np.mean(cond_data[c])
            c_sd = np.std(cond_data[c], ddof=1)
            threshold = c_mean + k * c_sd
            n_out = np.sum(cond_data[c] > threshold)
            pct = 100 * n_out / len(cond_data[c])
            print(f"    C{c}: threshold={threshold:>10,.0f}, {n_out:>3} outliers ({pct:>5.1f}%)")

    return cond_data, conditions


def part2_outlier_impact(cond_data, conditions):
    """Recompute per-condition means with and without outliers."""
    section_header("2. OUTLIER IMPACT ANALYSIS")

    print("Per-condition means WITH vs WITHOUT outliers (per-condition SD thresholds):\n")

    for k in [1, 2, 3]:
        print(f"--- +{k} SD cutoff ---")
        print(
            f"{'Condition':<12} {'Raw Mean':>12} {'Filtered Mean':>14} {'N Removed':>10} {'Change':>10}"
        )
        print("-" * 62)

        filtered_means = {}
        for c in conditions:
            raw_mean = np.mean(cond_data[c])
            c_mean = np.mean(cond_data[c])
            c_sd = np.std(cond_data[c], ddof=1)
            threshold = c_mean + k * c_sd
            filtered = cond_data[c][cond_data[c] <= threshold]
            filtered_mean = np.mean(filtered) if len(filtered) > 0 else 0
            n_removed = len(cond_data[c]) - len(filtered)
            change = 100 * (filtered_mean - raw_mean) / raw_mean
            filtered_means[c] = filtered_mean
            print(
                f"  C{c}       {raw_mean:>12,.1f} {filtered_mean:>14,.1f} {n_removed:>10} {change:>9.1f}%"
            )

        # Condition ordering
        raw_order = sorted(conditions, key=lambda c: np.mean(cond_data[c]))
        filt_order = sorted(conditions, key=lambda c: filtered_means[c])
        print(f"\n  Raw ordering (low to high):      {' < '.join(f'C{c}' for c in raw_order)}")
        print(f"  Filtered ordering (low to high): {' < '.join(f'C{c}' for c in filt_order)}")
        ordering_changed = raw_order != filt_order
        print(f"  Ordering changed: {'YES' if ordering_changed else 'No'}")

        # Feedback effect: C1 vs best treatment
        raw_c1 = np.mean(cond_data[1])
        best_raw = min(np.mean(cond_data[c]) for c in [2, 3, 4])
        filt_c1 = filtered_means[1]
        best_filt = min(filtered_means[c] for c in [2, 3, 4])
        raw_reduction = 100 * (raw_c1 - best_raw) / raw_c1
        filt_reduction = 100 * (filt_c1 - best_filt) / filt_c1
        print("\n  Feedback effect (C1 vs best treatment):")
        print(f"    Raw:      {raw_reduction:.1f}% reduction")
        print(f"    Filtered: {filt_reduction:.1f}% reduction")
        if filt_reduction > raw_reduction:
            print(
                f"    => Removing outliers STRENGTHENS the effect (+{filt_reduction - raw_reduction:.1f}pp)"
            )
        else:
            print(
                f"    => Removing outliers WEAKENS the effect ({filt_reduction - raw_reduction:.1f}pp)"
            )
        print()


def part3_winsorized(cond_data, conditions):
    """Winsorize at 95th and 99th percentile."""
    section_header("3. WINSORIZED ANALYSIS")

    all_tokens = np.concatenate([cond_data[c] for c in conditions])

    for pct in [99, 95]:
        threshold = np.percentile(all_tokens, pct)
        print(f"--- Winsorized at {pct}th percentile (cap at {threshold:,.0f}) ---")
        print(
            f"{'Condition':<12} {'Raw Mean':>12} {'Winsorized Mean':>16} {'N Capped':>10} {'Change':>10}"
        )
        print("-" * 64)

        for c in conditions:
            raw_mean = np.mean(cond_data[c])
            winsorized = np.clip(cond_data[c], None, threshold)
            win_mean = np.mean(winsorized)
            n_capped = np.sum(cond_data[c] > threshold)
            change = 100 * (win_mean - raw_mean) / raw_mean
            print(
                f"  C{c}       {raw_mean:>12,.1f} {win_mean:>16,.1f} {n_capped:>10} {change:>9.1f}%"
            )
        print()

    # Also per-condition winsorization
    print("--- Per-condition winsorization ---")
    for pct in [99, 95]:
        print(f"\n  {pct}th percentile (per-condition thresholds):")
        for c in conditions:
            threshold = np.percentile(cond_data[c], pct)
            raw_mean = np.mean(cond_data[c])
            winsorized = np.clip(cond_data[c], None, threshold)
            win_mean = np.mean(winsorized)
            n_capped = np.sum(cond_data[c] > threshold)
            print(
                f"    C{c}: cap={threshold:>10,.0f}, "
                f"raw={raw_mean:>10,.1f}, "
                f"win={win_mean:>10,.1f}, "
                f"n_capped={n_capped}"
            )


def part4_outlier_characterization(rows, cond_data):
    """Characterize the top 20 outlier episodes."""
    section_header("4. OUTLIER CHARACTERIZATION (Top 20 by total_tokens)")

    # Sort all episodes by total_tokens descending
    sorted_rows = sorted(rows, key=lambda r: r["total_tokens"], reverse=True)

    print(
        f"{'Rank':<5} {'EpID':>6} {'Cond':>5} {'Tokens':>10} {'Steps':>6} {'Preset':<16} {'Theme':<20} {'Task Title'}"
    )
    print("-" * 120)

    for i, r in enumerate(sorted_rows[:20]):
        title = r["task_title"][:40] if r["task_title"] else "N/A"
        preset = r["preset_id"] or "N/A"
        theme = r["task_theme"] or "N/A"
        print(
            f"{i + 1:<5} {r['episode_id']:>6} C{r['condition_id']:>4} "
            f"{r['total_tokens']:>10,} {r['step_count']:>6} "
            f"{preset:<16} {theme:<20} {title}"
        )

    # Concentration analysis
    print("\n--- Outlier concentration (top 20) ---")
    top20 = sorted_rows[:20]

    # By condition
    cond_counts = defaultdict(int)
    for r in top20:
        cond_counts[r["condition_id"]] += 1
    print("\n  By condition:")
    for c in sorted(cond_counts.keys()):
        print(f"    C{c}: {cond_counts[c]} episodes")

    # By task theme
    theme_counts = defaultdict(int)
    for r in top20:
        theme_counts[r["task_theme"]] += 1
    print("\n  By task theme:")
    for theme, count in sorted(theme_counts.items(), key=lambda x: -x[1]):
        print(f"    {theme}: {count}")

    # By preset
    preset_counts = defaultdict(int)
    for r in top20:
        preset_counts[r["preset_id"] or "None"] += 1
    print("\n  By preset:")
    for preset, count in sorted(preset_counts.items(), key=lambda x: -x[1]):
        print(f"    {preset}: {count}")

    # Step count stats for outliers vs non-outliers
    top20_steps = [r["step_count"] for r in top20 if r["step_count"] is not None]
    all_steps = [r["step_count"] for r in rows if r["step_count"] is not None]
    if top20_steps and all_steps:
        print("\n  Step count comparison:")
        print(
            f"    Top 20 outliers:  mean={np.mean(top20_steps):.1f}, median={np.median(top20_steps):.0f}"
        )
        print(
            f"    All episodes:     mean={np.mean(all_steps):.1f}, median={np.median(all_steps):.0f}"
        )

    # Extended: top 50 by condition
    print("\n--- Extended: top 50 episodes by condition ---")
    top50 = sorted_rows[:50]
    cond_counts_50 = defaultdict(int)
    for r in top50:
        cond_counts_50[r["condition_id"]] += 1
    for c in sorted(cond_counts_50.keys()):
        expected = 50 * len(cond_data[c]) / sum(len(cond_data[cc]) for cc in cond_data)
        print(f"  C{c}: {cond_counts_50[c]} (expected if uniform: {expected:.1f})")


def part5_sensitivity_table(rows, cond_data, conditions):
    """Sensitivity analysis table with p-values."""
    section_header("5. SENSITIVITY ANALYSIS TABLE")

    results = []

    # Raw
    raw_means = {c: np.mean(cond_data[c]) for c in conditions}
    t_stat, p_val = stats.ttest_ind(cond_data[1], cond_data[3], equal_var=False)
    c1_c3_diff = raw_means[1] - raw_means[3]
    results.append(
        ("Raw", raw_means, c1_c3_diff, p_val, {c: len(cond_data[c]) for c in conditions})
    )

    # SD cutoffs (per-condition thresholds, as paper describes)
    for k in [3, 2, 1]:
        filtered = {}
        for c in conditions:
            c_mean = np.mean(cond_data[c])
            c_sd = np.std(cond_data[c], ddof=1)
            threshold = c_mean + k * c_sd
            filtered[c] = cond_data[c][cond_data[c] <= threshold]

        filt_means = {c: np.mean(filtered[c]) for c in conditions}
        t_stat, p_val = stats.ttest_ind(filtered[1], filtered[3], equal_var=False)
        diff = filt_means[1] - filt_means[3]
        results.append(
            (f"+{k} SD", filt_means, diff, p_val, {c: len(filtered[c]) for c in conditions})
        )

    # Winsorized (global thresholds)
    all_tokens = np.concatenate([cond_data[c] for c in conditions])
    for pct in [99, 95]:
        threshold = np.percentile(all_tokens, pct)
        win_data = {c: np.clip(cond_data[c], None, threshold) for c in conditions}
        win_means = {c: np.mean(win_data[c]) for c in conditions}
        t_stat, p_val = stats.ttest_ind(win_data[1], win_data[3], equal_var=False)
        diff = win_means[1] - win_means[3]
        results.append(
            (f"Win {pct}%", win_means, diff, p_val, {c: len(win_data[c]) for c in conditions})
        )

    # Print table
    header = f"{'Cutoff':<10} {'C1 Mean':>10} {'C2 Mean':>10} {'C3 Mean':>10} {'C4 Mean':>10} {'C1-C3 Diff':>12} {'p-value':>12} {'C1 n':>6} {'C2 n':>6} {'C3 n':>6} {'C4 n':>6}"
    print(header)
    print("-" * len(header))

    for label, means, diff, p_val, ns in results:
        p_str = f"{p_val:.2e}" if p_val < 0.001 else f"{p_val:.4f}"
        print(
            f"{label:<10} {means[1]:>10,.1f} {means[2]:>10,.1f} "
            f"{means[3]:>10,.1f} {means[4]:>10,.1f} "
            f"{diff:>12,.1f} {p_str:>12} "
            f"{ns[1]:>6} {ns[2]:>6} {ns[3]:>6} {ns[4]:>6}"
        )

    # Also compute Cohen's d for each
    print("\n--- Effect sizes (Cohen's d, C1 vs C3) ---")
    print(f"{'Cutoff':<10} {'Cohen d':>10} {'Interpretation'}")
    print("-" * 40)

    # Raw
    d_raw = cohen_d(cond_data[1], cond_data[3])
    print(f"{'Raw':<10} {d_raw:>10.3f} {interpret_d(d_raw)}")

    for k in [3, 2, 1]:
        f1 = cond_data[1][cond_data[1] <= np.mean(cond_data[1]) + k * np.std(cond_data[1], ddof=1)]
        f3 = cond_data[3][cond_data[3] <= np.mean(cond_data[3]) + k * np.std(cond_data[3], ddof=1)]
        d = cohen_d(f1, f3)
        print(f"{f'+{k} SD':<10} {d:>10.3f} {interpret_d(d)}")

    all_t = np.concatenate([cond_data[c] for c in conditions])
    for pct in [99, 95]:
        threshold = np.percentile(all_t, pct)
        w1 = np.clip(cond_data[1], None, threshold)
        w3 = np.clip(cond_data[3], None, threshold)
        d = cohen_d(w1, w3)
        print(f"{f'Win {pct}%':<10} {d:>10.3f} {interpret_d(d)}")

    # Extended: C1 vs best treatment at each cutoff
    print("\n--- C1 vs BEST treatment (% reduction) ---")
    print(f"{'Cutoff':<10} {'C1 Mean':>10} {'Best Trt':>10} {'% Reduction':>12}")
    print("-" * 46)

    for label, means, diff, p_val, ns in results:
        best = min(means[c] for c in [2, 3, 4])
        reduction = 100 * (means[1] - best) / means[1]
        best_cond = min([2, 3, 4], key=lambda c: means[c])
        print(f"{label:<10} {means[1]:>10,.1f} {best:>10,.1f} {reduction:>11.1f}% (C{best_cond})")


def cohen_d(group1, group2):
    """Compute Cohen's d (pooled SD)."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_sd == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_sd


def interpret_d(d):
    """Interpret Cohen's d magnitude."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def part6_additional_diagnostics(cond_data, conditions):
    """Additional diagnostic information."""
    section_header("6. ADDITIONAL DIAGNOSTICS")

    # Normality tests
    print("Shapiro-Wilk normality test (p < 0.05 => reject normality):")
    for c in conditions:
        # Shapiro-Wilk has n<=5000 limit, use first 300
        stat, p = stats.shapiro(cond_data[c][:300])
        print(f"  C{c}: W={stat:.4f}, p={p:.2e} {'*** NON-NORMAL' if p < 0.05 else ''}")

    # Kruskal-Wallis (non-parametric alternative to ANOVA)
    print("\nKruskal-Wallis H-test (non-parametric):")
    h_stat, p_val = stats.kruskal(*[cond_data[c] for c in conditions])
    print(f"  H={h_stat:.2f}, p={p_val:.2e}")

    # Mann-Whitney U for C1 vs each treatment
    print("\nMann-Whitney U tests (C1 vs each treatment):")
    for c in [2, 3, 4]:
        u_stat, p_val = stats.mannwhitneyu(cond_data[1], cond_data[c], alternative="greater")
        print(f"  C1 vs C{c}: U={u_stat:.0f}, p={p_val:.2e}")

    # Percentile comparison
    print("\nPercentile comparison:")
    print(f"{'Percentile':<12}", end="")
    for c in conditions:
        print(f"{'C' + str(c):>10}", end="")
    print()
    print("-" * 52)
    for pctl in [25, 50, 75, 90, 95, 99]:
        print(f"{pctl:>4}th       ", end="")
        for c in conditions:
            val = np.percentile(cond_data[c], pctl)
            print(f"{val:>10,.0f}", end="")
        print()


if __name__ == "__main__":
    print("=" * 80)
    print("  OUTLIER ANALYSIS — v3 Experiment Data")
    print("  Database: experiment_v3.db")
    print("=" * 80)

    rows = load_data()
    print(f"\nLoaded {len(rows)} episodes")

    cond_data, conditions = part1_identify_outliers(rows)
    part2_outlier_impact(cond_data, conditions)
    part3_winsorized(cond_data, conditions)
    part4_outlier_characterization(rows, cond_data)
    part5_sensitivity_table(rows, cond_data, conditions)
    part6_additional_diagnostics(cond_data, conditions)

    print(f"\n{'=' * 80}")
    print("  ANALYSIS COMPLETE")
    print(f"{'=' * 80}")
