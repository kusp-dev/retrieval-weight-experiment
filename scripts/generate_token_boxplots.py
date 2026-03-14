"""Generate separate token consumption boxplots for v2 and v3 experiments."""

import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COLORS = {1: "#E74C3C", 2: "#5B8BD6", 3: "#2ECC71", 4: "#9B59B6"}
LABELS = {1: "C1 (Control)", 2: "C2 (Likert)", 3: "C3 (Full)", 4: "C4 (Anchor)"}
FIG_DIR = Path("results/figures")


def get_tokens_by_condition(db_path: str) -> dict[int, list[int]]:
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT condition_id, total_tokens FROM episodes ORDER BY condition_id"
    ).fetchall()
    conn.close()
    tokens: dict[int, list[int]] = {}
    for cid, tok in rows:
        tokens.setdefault(cid, []).append(tok)
    return tokens


def make_boxplot(
    tokens: dict[int, list[int]],
    title: str,
    output_path: Path,
    ylim: tuple[int, int] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    conditions = sorted(tokens.keys())
    data = [tokens[c] for c in conditions]
    positions = list(range(1, len(conditions) + 1))

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showfliers=True,
        flierprops=dict(marker="o", markersize=3, alpha=0.4, color="#aaaaaa"),
        whiskerprops=dict(color="white", linewidth=1.2),
        capprops=dict(color="white", linewidth=1.2),
        medianprops=dict(color="white", linewidth=2),
    )

    for patch, cid in zip(bp["boxes"], conditions):
        patch.set_facecolor(COLORS[cid])
        patch.set_edgecolor("white")
        patch.set_alpha(0.85)

    # Add mean diamonds
    means = [np.mean(tokens[c]) for c in conditions]
    ax.scatter(
        positions,
        means,
        marker="D",
        color="white",
        s=60,
        zorder=5,
        label="Mean",
    )

    # Labels
    ax.set_xticks(positions)
    xlabels = []
    for c in conditions:
        n = len(tokens[c])
        xlabels.append(f"{LABELS[c]}\nn={n}")
    ax.set_xticklabels(xlabels, color="white", fontsize=10)

    ax.set_ylabel("Total Tokens", color="white", fontsize=12)
    ax.set_title(title, color="white", fontsize=14, fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
        spine.set_linewidth(0.5)

    ax.legend(loc="upper right", facecolor="#1a1a2e", edgecolor="white", labelcolor="white")

    if ylim is not None:
        ax.set_ylim(ylim)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor="#1a1a2e", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")

    # Print summary stats
    for c in conditions:
        d = tokens[c]
        print(
            f"  {LABELS[c]}: mean={np.mean(d):.0f}, "
            f"median={np.median(d):.0f}, sd={np.std(d):.0f}, n={len(d)}"
        )


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # V2 boxplot
    print("\n=== Experiment 1 (v2) ===")
    v2_tokens = get_tokens_by_condition("experiment.db")
    make_boxplot(
        v2_tokens,
        "Token Consumption by Condition (Experiment 1)",
        FIG_DIR / "fig1_v2_token_boxplot.png",
        ylim=(0, 10_000),
    )

    # V3 boxplot
    print("\n=== Experiment 2 (v3) ===")
    v3_tokens = get_tokens_by_condition("experiment_v3.db")
    make_boxplot(
        v3_tokens,
        "Token Consumption by Condition (Experiment 2)",
        FIG_DIR / "fig1_v3_token_boxplot.png",
        ylim=(0, 12_500),
    )


if __name__ == "__main__":
    main()
