#!/usr/bin/env python3
"""
Per-condition I/O showcase analysis for the research paper.

Produces:
  - Per-condition input vs output token breakdown (mean, median, SD)
  - I/O ratio by condition, difficulty, and theme
  - Token distribution shape (skewness, kurtosis)
  - Showcase task deep dives (5 tasks across 4 conditions)
  - Publication-quality figures (stacked bar, heatmap, showcase comparison)

Usage:
  uv run python scripts/per_condition_analysis.py
"""

import sqlite3
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "experiment_v3.db"
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants (matching analyze_experiment.py) ─────────────────────────────
COND_NAMES = {
    1: "C1 Control",
    2: "C2 Dim Feedback",
    3: "C3 Full System",
    4: "C4 Qualitative",
}
COND_SHORT = {1: "C1", 2: "C2", 3: "C3", 4: "C4"}
COND_COLORS = {
    1: "#9ca3af",
    2: "#60a5fa",
    3: "#34d399",
    4: "#a78bfa",
}

# Lighter variants for input portion of stacked bars
COND_COLORS_INPUT = {
    1: "#d1d5db",
    2: "#93c5fd",
    3: "#6ee7b7",
    4: "#c4b5fd",
}

# Dark theme
DARK_BG = "#0a0a0a"
DARK_FG = "#ededed"
DARK_GRID = "#1f1f1f"
DARK_ACCENT = "#2a2a2a"

SHOWCASE_TASKS = ["task_023", "task_286", "task_265", "task_175", "task_120"]

THEME_LABELS = {
    "agent_architectures": "Agent Arch",
    "evaluation_methods": "Eval Methods",
    "ml_fundamentals": "ML Fund",
    "multi_agent_systems": "Multi-Agent",
    "retrieval_systems": "Retrieval Sys",
}


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


def get_db() -> sqlite3.Connection:
    if not DB_PATH.exists():
        print(f"[ERROR] Database not found: {DB_PATH}")
        sys.exit(1)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


# ── Statistical helpers (no scipy dependency) ─────────────────────────────


def skewness(arr):
    """Sample skewness (moment-based, adjusted for bias)."""
    n = len(arr)
    if n < 3:
        return float("nan")
    mean = np.mean(arr)
    m2 = np.mean((arr - mean) ** 2)
    m3 = np.mean((arr - mean) ** 3)
    if m2 == 0:
        return 0.0
    # Adjusted Fisher-Pearson (same as scipy.stats.skew with bias=False)
    G1 = m3 / (m2**1.5)
    return G1 * np.sqrt(n * (n - 1)) / (n - 2)


def kurtosis_excess(arr):
    """Excess kurtosis (moment-based, normal = 0)."""
    n = len(arr)
    if n < 4:
        return float("nan")
    mean = np.mean(arr)
    m2 = np.mean((arr - mean) ** 2)
    m4 = np.mean((arr - mean) ** 4)
    if m2 == 0:
        return 0.0
    # Moment-based excess kurtosis: m4/m2^2 - 3
    return m4 / (m2**2) - 3.0


# ── Analysis functions ─────────────────────────────────────────────────────


def per_condition_summary(conn):
    """Print per-condition I/O breakdown: mean, median, SD, I/O ratio."""
    rows = conn.execute("""
        SELECT condition_id, input_tokens, output_tokens, total_tokens
        FROM episodes
        WHERE completed_at IS NOT NULL
        ORDER BY condition_id
    """).fetchall()

    data = {c: {"input": [], "output": [], "total": []} for c in range(1, 5)}
    for r in rows:
        c = r["condition_id"]
        data[c]["input"].append(r["input_tokens"])
        data[c]["output"].append(r["output_tokens"])
        data[c]["total"].append(r["total_tokens"])

    print("=" * 90)
    print("PER-CONDITION I/O TOKEN BREAKDOWN")
    print("=" * 90)
    header = (
        f"{'Cond':<18} {'N':>4} | {'Mean In':>8} {'Med In':>8} "
        f"{'SD In':>8} | {'Mean Out':>9} {'Med Out':>9} "
        f"{'SD Out':>8} | {'I/O Ratio':>9}"
    )
    print(header)
    print("-" * 90)

    summary = {}
    for c in range(1, 5):
        inp = np.array(data[c]["input"])
        out = np.array(data[c]["output"])
        tot = np.array(data[c]["total"])
        io_ratio = inp / tot

        s = {
            "n": len(inp),
            "mean_in": np.mean(inp),
            "median_in": np.median(inp),
            "sd_in": np.std(inp, ddof=1),
            "mean_out": np.mean(out),
            "median_out": np.median(out),
            "sd_out": np.std(out, ddof=1),
            "mean_io_ratio": np.mean(io_ratio),
            "inp": inp,
            "out": out,
            "tot": tot,
        }
        summary[c] = s

        print(
            f"{COND_NAMES[c]:<18} {s['n']:>4} | "
            f"{s['mean_in']:>8.1f} {s['median_in']:>8.1f} {s['sd_in']:>8.1f} | "
            f"{s['mean_out']:>9.1f} {s['median_out']:>9.1f} {s['sd_out']:>8.1f} | "
            f"{s['mean_io_ratio']:>9.4f}"
        )

    # Delta vs C1
    print()
    print("Delta vs C1 (Control):")
    for c in [2, 3, 4]:
        d_in = summary[c]["mean_in"] - summary[1]["mean_in"]
        d_out = summary[c]["mean_out"] - summary[1]["mean_out"]
        d_tot = np.mean(summary[c]["tot"]) - np.mean(summary[1]["tot"])
        pct = d_tot / np.mean(summary[1]["tot"]) * 100
        print(
            f"  {COND_SHORT[c]}: input {d_in:+.1f}, output {d_out:+.1f}, "
            f"total {d_tot:+.1f} ({pct:+.1f}%)"
        )

    return summary


def distribution_shape(summary):
    """Skewness and kurtosis per condition for input and output tokens."""
    print()
    print("=" * 80)
    print("TOKEN DISTRIBUTION SHAPE (Skewness & Excess Kurtosis)")
    print("=" * 80)
    header = (
        f"{'Cond':<18} | {'Skew(In)':>10} {'Kurt(In)':>10} | {'Skew(Out)':>10} {'Kurt(Out)':>10}"
    )
    print(header)
    print("-" * 80)

    shape_data = {}
    for c in range(1, 5):
        sk_in = skewness(summary[c]["inp"])
        ku_in = kurtosis_excess(summary[c]["inp"])
        sk_out = skewness(summary[c]["out"])
        ku_out = kurtosis_excess(summary[c]["out"])
        shape_data[c] = {
            "skew_in": sk_in,
            "kurt_in": ku_in,
            "skew_out": sk_out,
            "kurt_out": ku_out,
        }
        print(
            f"{COND_NAMES[c]:<18} | {sk_in:>10.3f} {ku_in:>10.3f} | {sk_out:>10.3f} {ku_out:>10.3f}"
        )

    # Interpretation
    print()
    for c in range(1, 5):
        sd = shape_data[c]
        interp = []
        if sd["skew_out"] > 1.0:
            interp.append("heavy right tail (output)")
        elif sd["skew_out"] > 0.5:
            interp.append("moderate right skew (output)")
        if sd["kurt_out"] > 3.0:
            interp.append("leptokurtic output (extreme outliers)")
        elif sd["kurt_out"] > 1.0:
            interp.append("somewhat heavy-tailed output")
        if interp:
            print(f"  {COND_SHORT[c]}: {', '.join(interp)}")

    return shape_data


def io_ratio_by_difficulty(conn):
    """I/O ratio breakdown by difficulty level and condition."""
    rows = conn.execute("""
        SELECT t.difficulty, e.condition_id, e.input_tokens, e.total_tokens
        FROM episodes e
        JOIN tasks t ON e.task_id = t.task_id
        WHERE e.completed_at IS NOT NULL
        ORDER BY t.difficulty, e.condition_id
    """).fetchall()

    data = {}
    for r in rows:
        key = (r["difficulty"], r["condition_id"])
        if key not in data:
            data[key] = []
        data[key].append(r["input_tokens"] / r["total_tokens"])

    print()
    print("=" * 70)
    print("I/O RATIO BY DIFFICULTY x CONDITION")
    print("=" * 70)
    difficulties = ["easy", "medium", "hard"]
    header = f"{'Difficulty':<10} | " + " | ".join(f"{COND_SHORT[c]:>12}" for c in range(1, 5))
    print(header)
    print("-" * 70)

    diff_data = {}
    for d in difficulties:
        row_vals = []
        for c in range(1, 5):
            vals = data.get((d, c), [])
            m = np.mean(vals) if vals else 0
            row_vals.append(m)
        diff_data[d] = row_vals
        print(f"{d:<10} | " + " | ".join(f"{v:>12.4f}" for v in row_vals))

    return diff_data


def io_ratio_by_theme(conn):
    """I/O ratio breakdown by theme and condition — returns data for heatmap."""
    rows = conn.execute("""
        SELECT t.theme, e.condition_id,
               e.input_tokens, e.output_tokens, e.total_tokens
        FROM episodes e
        JOIN tasks t ON e.task_id = t.task_id
        WHERE e.completed_at IS NOT NULL
    """).fetchall()

    data = {}
    for r in rows:
        key = (r["theme"], r["condition_id"])
        if key not in data:
            data[key] = {"io_ratios": [], "inputs": [], "outputs": []}
        data[key]["io_ratios"].append(r["input_tokens"] / r["total_tokens"])
        data[key]["inputs"].append(r["input_tokens"])
        data[key]["outputs"].append(r["output_tokens"])

    themes = sorted({k[0] for k in data.keys()})

    print()
    print("=" * 80)
    print("I/O RATIO BY THEME x CONDITION")
    print("=" * 80)
    header = f"{'Theme':<25} | " + " | ".join(f"{COND_SHORT[c]:>9}" for c in range(1, 5))
    print(header)
    print("-" * 80)

    theme_data = {}
    for theme in themes:
        row_vals = []
        for c in range(1, 5):
            vals = data.get((theme, c), {}).get("io_ratios", [])
            m = np.mean(vals) if vals else 0
            row_vals.append(m)
        theme_data[theme] = row_vals
        label = THEME_LABELS.get(theme, theme)
        print(f"{label:<25} | " + " | ".join(f"{v:>9.4f}" for v in row_vals))

    # Also print output token means by theme
    print()
    print("Mean OUTPUT tokens by theme x condition:")
    header2 = f"{'Theme':<25} | " + " | ".join(f"{COND_SHORT[c]:>9}" for c in range(1, 5))
    print(header2)
    print("-" * 80)
    for theme in themes:
        row_vals = []
        for c in range(1, 5):
            vals = data.get((theme, c), {}).get("outputs", [])
            m = np.mean(vals) if vals else 0
            row_vals.append(m)
        label = THEME_LABELS.get(theme, theme)
        print(f"{label:<25} | " + " | ".join(f"{v:>9.1f}" for v in row_vals))

    return theme_data, themes


def showcase_tasks(conn):
    """Deep dive into 5 showcase tasks across all 4 conditions."""
    # Task metadata
    task_meta = {}
    for r in conn.execute(
        """
        SELECT task_id, theme, title, difficulty, ground_truth_skills
        FROM tasks WHERE task_id IN ({})
    """.format(",".join(f"'{t}'" for t in SHOWCASE_TASKS))
    ).fetchall():
        task_meta[r["task_id"]] = dict(r)

    # Episode data
    episodes = conn.execute(
        """
        SELECT e.episode_id, e.condition_id, e.task_id,
               e.input_tokens, e.output_tokens, e.total_tokens,
               e.step_count, e.success, e.preset_id, e.duration_ms
        FROM episodes e
        WHERE e.task_id IN ({})
        AND e.completed_at IS NOT NULL
        ORDER BY e.task_id, e.condition_id
    """.format(",".join(f"'{t}'" for t in SHOWCASE_TASKS))
    ).fetchall()

    # Retrieval results per episode
    ep_ids = [e["episode_id"] for e in episodes]
    retrieval = {}
    if ep_ids:
        for r in conn.execute(
            """
            SELECT rr.episode_id, rr.skill_id, rr.rank, rr.final_score,
                   rr.is_ground_truth, rr.recency_score, rr.importance_score,
                   rr.relevance_score
            FROM retrieval_results rr
            WHERE rr.episode_id IN ({})
            ORDER BY rr.episode_id, rr.rank
        """.format(",".join(str(eid) for eid in ep_ids))
        ).fetchall():
            eid = r["episode_id"]
            if eid not in retrieval:
                retrieval[eid] = []
            retrieval[eid].append(dict(r))

    print()
    print("=" * 100)
    print("SHOWCASE TASK DEEP DIVES (5 tasks x 4 conditions)")
    print("=" * 100)

    showcase_data = {}

    for task_id in SHOWCASE_TASKS:
        meta = task_meta.get(task_id, {})
        theme_label = THEME_LABELS.get(meta.get("theme", ""), meta.get("theme", ""))
        print(f"\n--- {task_id}: {meta.get('title', 'N/A')} ---")
        print(f"    Theme: {theme_label} | Difficulty: {meta.get('difficulty', 'N/A')}")
        if meta.get("ground_truth_skills"):
            print(f"    Ground truth: {meta['ground_truth_skills']}")

        task_episodes = [e for e in episodes if e["task_id"] == task_id]
        task_data = {}

        for ep in task_episodes:
            c = ep["condition_id"]
            eid = ep["episode_id"]
            skills = retrieval.get(eid, [])
            skill_ids = [s["skill_id"] for s in skills]
            gt_hits = sum(1 for s in skills if s["is_ground_truth"])

            task_data[c] = {
                "input": ep["input_tokens"],
                "output": ep["output_tokens"],
                "total": ep["total_tokens"],
                "steps": ep["step_count"],
                "preset": ep["preset_id"],
                "skills": skill_ids,
                "gt_hits": gt_hits,
                "io_ratio": ep["input_tokens"] / ep["total_tokens"] if ep["total_tokens"] else 0,
            }

            print(
                f"    {COND_SHORT[c]}: in={ep['input_tokens']:>5}, "
                f"out={ep['output_tokens']:>5}, total={ep['total_tokens']:>6}, "
                f"steps={ep['step_count']}, preset={ep['preset_id']:<20} "
                f"GT hits={gt_hits}/5  skills={', '.join(skill_ids[:5])}"
            )

        # Output length variation
        outputs = [task_data[c]["output"] for c in sorted(task_data.keys())]
        if len(outputs) > 1:
            out_range = max(outputs) - min(outputs)
            out_ratio = max(outputs) / min(outputs) if min(outputs) > 0 else float("inf")
            print(f"    -> Output range: {out_range} tokens ({out_ratio:.1f}x ratio max/min)")

        showcase_data[task_id] = task_data

    return showcase_data, task_meta


# ── Figures ────────────────────────────────────────────────────────────────


def fig_stacked_bar(summary):
    """Fig: Stacked bar chart — input vs output tokens by condition."""
    fig, ax = plt.subplots(figsize=(10, 6))

    conditions = [1, 2, 3, 4]
    x = np.arange(len(conditions))
    width = 0.55

    input_means = [summary[c]["mean_in"] for c in conditions]
    output_means = [summary[c]["mean_out"] for c in conditions]

    ax.bar(
        x,
        input_means,
        width,
        label="Input tokens",
        color=[COND_COLORS_INPUT[c] for c in conditions],
        edgecolor=DARK_GRID,
        linewidth=0.5,
    )
    ax.bar(
        x,
        output_means,
        width,
        bottom=input_means,
        label="Output tokens",
        color=[COND_COLORS[c] for c in conditions],
        edgecolor=DARK_GRID,
        linewidth=0.5,
    )

    # Annotate totals on top
    for i, c in enumerate(conditions):
        total = input_means[i] + output_means[i]
        ax.text(
            i,
            total + 80,
            f"{total:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=DARK_FG,
        )
        # I/O ratio inside bar
        ratio = input_means[i] / total
        ax.text(
            i,
            input_means[i] / 2,
            f"{ratio:.1%}",
            ha="center",
            va="center",
            fontsize=9,
            color="#333333",
            fontweight="bold",
        )

    ax.set_xlabel("Condition")
    ax.set_ylabel("Mean Tokens per Episode")
    ax.set_title("Input vs Output Token Breakdown by Condition")
    ax.set_xticks(x)
    ax.set_xticklabels([COND_NAMES[c] for c in conditions], fontsize=10)
    ax.legend(loc="upper right")
    ax.set_ylim(0, max(input_means[i] + output_means[i] for i in range(4)) * 1.12)

    # Add C1 baseline reference line
    c1_total = input_means[0] + output_means[0]
    ax.axhline(y=c1_total, color=COND_COLORS[1], linestyle="--", alpha=0.5, linewidth=1)
    ax.text(3.4, c1_total + 50, "C1 baseline", fontsize=8, color=COND_COLORS[1], alpha=0.7)

    fig.tight_layout()
    path = FIG_DIR / "figIO1_stacked_bar_tokens.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"\n  [Saved] {path}")


def fig_theme_heatmap(theme_data, themes):
    """Fig: I/O ratio heatmap by theme x condition."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Build matrix: rows=themes, cols=conditions
    matrix = np.array([theme_data[t] for t in themes])
    theme_labels = [THEME_LABELS.get(t, t) for t in themes]
    cond_labels = [COND_SHORT[c] for c in range(1, 5)]

    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0.30, vmax=0.55)
    fig.colorbar(im, ax=ax, label="I/O Ratio (input / total)")

    # Tick labels
    ax.set_xticks(np.arange(len(cond_labels)))
    ax.set_yticks(np.arange(len(theme_labels)))
    ax.set_xticklabels(cond_labels, fontsize=11)
    ax.set_yticklabels(theme_labels, fontsize=10)

    # Annotate cells
    for i in range(len(themes)):
        for j in range(4):
            val = matrix[i, j]
            text_color = "#333333" if val > 0.45 else DARK_FG
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

    ax.set_title("I/O Ratio by Theme and Condition")
    ax.set_xlabel("Condition")
    ax.set_ylabel("Theme")

    fig.tight_layout()
    path = FIG_DIR / "figIO2_theme_heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [Saved] {path}")


def fig_showcase_comparison(showcase_data, task_meta):
    """Fig: Side-by-side bars for each showcase task across conditions."""
    n_tasks = len(SHOWCASE_TASKS)
    fig, axes = plt.subplots(1, n_tasks, figsize=(16, 6), sharey=False)

    for idx, task_id in enumerate(SHOWCASE_TASKS):
        ax = axes[idx]
        td = showcase_data.get(task_id, {})
        meta = task_meta.get(task_id, {})

        conditions = sorted(td.keys())
        x = np.arange(len(conditions))
        width = 0.6

        inputs = [td[c]["input"] for c in conditions]
        outputs = [td[c]["output"] for c in conditions]

        ax.bar(
            x,
            inputs,
            width,
            color=[COND_COLORS_INPUT[c] for c in conditions],
            edgecolor=DARK_GRID,
            linewidth=0.5,
            label="Input" if idx == 0 else None,
        )
        ax.bar(
            x,
            outputs,
            width,
            bottom=inputs,
            color=[COND_COLORS[c] for c in conditions],
            edgecolor=DARK_GRID,
            linewidth=0.5,
            label="Output" if idx == 0 else None,
        )

        # Annotate totals
        for i, c in enumerate(conditions):
            total = inputs[i] + outputs[i]
            ax.text(
                i, total + 100, f"{total:,}", ha="center", va="bottom", fontsize=7, color=DARK_FG
            )

        # Title
        theme_label = THEME_LABELS.get(meta.get("theme", ""), "")
        diff = meta.get("difficulty", "")[0].upper() if meta.get("difficulty") else ""
        ax.set_title(f"{task_id}\n{diff} | {theme_label}", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([COND_SHORT[c] for c in conditions], fontsize=9)
        ax.tick_params(axis="y", labelsize=8)

        if idx == 0:
            ax.set_ylabel("Tokens", fontsize=10)

    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Showcase Task Token Profiles (5 Tasks x 4 Conditions)", fontsize=13, y=1.02)
    fig.tight_layout()

    path = FIG_DIR / "figIO3_showcase_comparison.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {path}")


def fig_io_ratio_by_difficulty(conn):
    """Fig: Grouped bar chart of I/O ratio by difficulty and condition."""
    rows = conn.execute("""
        SELECT t.difficulty, e.condition_id, e.input_tokens, e.total_tokens
        FROM episodes e
        JOIN tasks t ON e.task_id = t.task_id
        WHERE e.completed_at IS NOT NULL
    """).fetchall()

    data = {}
    for r in rows:
        key = (r["difficulty"], r["condition_id"])
        if key not in data:
            data[key] = []
        data[key].append(r["input_tokens"] / r["total_tokens"])

    difficulties = ["easy", "medium", "hard"]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(difficulties))
    width = 0.18
    offsets = [-(1.5 * width), -(0.5 * width), (0.5 * width), (1.5 * width)]

    for i, c in enumerate(range(1, 5)):
        vals = [np.mean(data.get((d, c), [0])) for d in difficulties]
        ax.bar(
            x + offsets[i],
            vals,
            width,
            label=COND_NAMES[c],
            color=COND_COLORS[c],
            edgecolor=DARK_GRID,
            linewidth=0.5,
        )
        for j, v in enumerate(vals):
            ax.text(
                x[j] + offsets[i],
                v + 0.005,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color=DARK_FG,
            )

    ax.set_xlabel("Difficulty")
    ax.set_ylabel("Mean I/O Ratio (input / total)")
    ax.set_title("I/O Ratio by Difficulty Level and Condition")
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in difficulties])
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0.28, 0.60)

    fig.tight_layout()
    path = FIG_DIR / "figIO4_difficulty_io_ratio.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [Saved] {path}")


# ── Skill overlap analysis for showcase tasks ──────────────────────────────


def showcase_skill_overlap(conn):
    """Compute Jaccard overlap of retrieved skills across conditions for showcase tasks."""
    print()
    print("=" * 80)
    print("SHOWCASE TASK SKILL OVERLAP (Jaccard)")
    print("=" * 80)

    for task_id in SHOWCASE_TASKS:
        episodes = conn.execute(
            """
            SELECT e.episode_id, e.condition_id
            FROM episodes e
            WHERE e.task_id = ? AND e.completed_at IS NOT NULL
            ORDER BY e.condition_id
        """,
            (task_id,),
        ).fetchall()

        cond_skills = {}
        for ep in episodes:
            skills = conn.execute(
                """
                SELECT skill_id FROM retrieval_results
                WHERE episode_id = ? ORDER BY rank
            """,
                (ep["episode_id"],),
            ).fetchall()
            cond_skills[ep["condition_id"]] = set(r["skill_id"] for r in skills)

        print(f"\n  {task_id}:")
        # Pairwise Jaccard
        for ci in range(1, 5):
            for cj in range(ci + 1, 5):
                si = cond_skills.get(ci, set())
                sj = cond_skills.get(cj, set())
                union = si | sj
                inter = si & sj
                jacc = len(inter) / len(union) if union else 0
                print(
                    f"    {COND_SHORT[ci]} vs {COND_SHORT[cj]}: Jaccard={jacc:.3f} "
                    f"(shared: {sorted(inter)}, |union|={len(union)})"
                )


# ── Main ───────────────────────────────────────────────────────────────────


def main():
    setup_dark_theme()
    conn = get_db()

    print("\n" + "=" * 90)
    print("  PER-CONDITION I/O SHOWCASE ANALYSIS")
    print("  Database:", DB_PATH)
    print("=" * 90)

    # 1. Per-condition summary
    summary = per_condition_summary(conn)

    # 2. Distribution shape
    shape_data = distribution_shape(summary)

    # 3. I/O ratio by difficulty
    io_ratio_by_difficulty(conn)

    # 4. I/O ratio by theme
    theme_data, themes = io_ratio_by_theme(conn)

    # 5. Showcase task deep dives
    showcase_data, task_meta = showcase_tasks(conn)

    # 6. Skill overlap for showcase tasks
    showcase_skill_overlap(conn)

    # 7. Generate figures
    print()
    print("=" * 60)
    print("GENERATING FIGURES")
    print("=" * 60)
    fig_stacked_bar(summary)
    fig_theme_heatmap(theme_data, themes)
    fig_showcase_comparison(showcase_data, task_meta)
    fig_io_ratio_by_difficulty(conn)

    # 8. Key findings summary
    print()
    print("=" * 90)
    print("KEY FINDINGS SUMMARY")
    print("=" * 90)

    # Output reduction
    c1_out = summary[1]["mean_out"]
    for c in [2, 3, 4]:
        pct = (summary[c]["mean_out"] - c1_out) / c1_out * 100
        print(f"  {COND_SHORT[c]} output vs C1: {pct:+.1f}%")

    # Input increase
    c1_in = summary[1]["mean_in"]
    for c in [2, 3, 4]:
        pct = (summary[c]["mean_in"] - c1_in) / c1_in * 100
        print(f"  {COND_SHORT[c]} input vs C1: {pct:+.1f}%")

    # Theme with largest I/O ratio gap
    max_gap = 0
    max_theme = ""
    for t in themes:
        gap = max(theme_data[t]) - min(theme_data[t])
        if gap > max_gap:
            max_gap = gap
            max_theme = t
    print(
        f"\n  Largest I/O ratio gap across conditions: {THEME_LABELS.get(max_theme, max_theme)} "
        f"(gap={max_gap:.4f})"
    )

    # Showcase task with largest output range
    max_range = 0
    max_task = ""
    for tid, td in showcase_data.items():
        outputs = [td[c]["output"] for c in td]
        r = max(outputs) - min(outputs)
        if r > max_range:
            max_range = r
            max_task = tid
    print(f"  Showcase task with largest output range: {max_task} ({max_range} tokens)")

    # Distribution insight
    for c in range(1, 5):
        sk = shape_data[c]["skew_out"]
        ku = shape_data[c]["kurt_out"]
        print(f"  {COND_SHORT[c]} output distribution: skew={sk:.3f}, excess kurtosis={ku:.3f}")

    print()
    print("Done. Figures saved to:", FIG_DIR)

    conn.close()


if __name__ == "__main__":
    main()
