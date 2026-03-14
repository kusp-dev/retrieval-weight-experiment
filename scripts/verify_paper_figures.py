#!/usr/bin/env python3
"""Verify that paper figures match committed data by regenerating and comparing.

Regenerates all figures via analyze_experiment.py, then compares the output
against the committed paper/figures/ versions. Reports any discrepancies.

This is a regression check to ensure paper figures are reproducible.

Usage:
    cd ~/retrieval-weight-experiment && uv run python scripts/verify_paper_figures.py
"""

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER_FIGS = ROOT / "paper" / "figures"
RESULTS_FIGS = ROOT / "results" / "figures"

# Figures that should be in paper/figures/
EXPECTED_FIGURES = [
    "fig1_reward_boxplot.png",
    "fig2_convergence_curves.png",
    "fig3_theme_heatmap.png",
    "fig4_step_efficiency.png",
    "fig5_cumulative_regret.png",
    "fig6_ground_truth_hit_rate.png",
    "fig7_jaccard_trajectory.png",
    "fig8_multistep_decomposition.png",
    "figA1_skill_retrieval_heatmap.png",
    "figA2_token_trend.png",
    "figA4_jaccard_vs_tokens.png",
    "figA6_rating_stability.png",
    "figA7_bandit_posteriors.png",
    "figA8_difficulty_comparison.png",
    "figA9_domain_heatmap.png",
]


def file_hash(path: Path) -> str:
    """SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def file_size(path: Path) -> int:
    """File size in bytes."""
    return path.stat().st_size


def main():
    print("=" * 80)
    print("Paper Figure Verification")
    print("=" * 80)

    # Step 1: Check what's in paper/figures/
    print("\n[1/4] Checking paper/figures/ directory...")
    if not PAPER_FIGS.exists():
        print(f"  ERROR: {PAPER_FIGS} does not exist!")
        sys.exit(1)

    paper_files = sorted(f.name for f in PAPER_FIGS.iterdir() if f.suffix == ".png")
    print(f"  Found {len(paper_files)} figures in paper/figures/")
    for f in paper_files:
        size = file_size(PAPER_FIGS / f) / 1024
        print(f"    {f}: {size:.1f} KB")

    # Check for missing expected figures
    missing = [f for f in EXPECTED_FIGURES if f not in paper_files]
    if missing:
        print(f"\n  WARNING: {len(missing)} expected figures missing from paper/figures/:")
        for f in missing:
            print(f"    - {f}")

    extra = [f for f in paper_files if f not in EXPECTED_FIGURES]
    if extra:
        print(f"\n  Note: {len(extra)} extra figures in paper/figures/ (not in expected list):")
        for f in extra:
            print(f"    + {f}")

    # Step 2: Record current results/figures/ state
    print("\n[2/4] Recording current results/figures/ state...")
    if RESULTS_FIGS.exists():
        pre_regen_hashes = {}
        for f in RESULTS_FIGS.iterdir():
            if f.suffix == ".png":
                pre_regen_hashes[f.name] = file_hash(f)
        print(f"  {len(pre_regen_hashes)} existing figures in results/figures/")
    else:
        pre_regen_hashes = {}
        print("  results/figures/ does not exist yet")

    # Step 3: Regenerate figures
    print("\n[3/4] Regenerating figures via analyze_experiment.py...")
    print("  (This may take a minute...)\n")

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "analyze_experiment.py")],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )

    if result.returncode != 0:
        print(f"  ERROR: analyze_experiment.py failed (exit code {result.returncode})")
        print(f"  STDERR: {result.stderr[:500]}")
        # Try to continue anyway if some figures were generated
        if not RESULTS_FIGS.exists() or not list(RESULTS_FIGS.glob("*.png")):
            print("  No figures generated. Cannot compare.")
            sys.exit(1)
    else:
        print("  Regeneration complete.")

    # Step 4: Compare
    print("\n[4/4] Comparing regenerated figures to paper/figures/...")
    print()

    regen_files = {f.name: f for f in RESULTS_FIGS.iterdir() if f.suffix == ".png"}
    print(f"  Regenerated {len(regen_files)} figures in results/figures/")

    discrepancies = []
    matches = []
    missing_regen = []

    for fig_name in EXPECTED_FIGURES:
        paper_path = PAPER_FIGS / fig_name
        regen_path = RESULTS_FIGS / fig_name

        if not paper_path.exists():
            discrepancies.append(
                {
                    "figure": fig_name,
                    "issue": "missing from paper/figures/",
                    "severity": "HIGH",
                }
            )
            continue

        if not regen_path.exists():
            missing_regen.append(fig_name)
            discrepancies.append(
                {
                    "figure": fig_name,
                    "issue": "not regenerated (script may have failed for this figure)",
                    "severity": "HIGH",
                }
            )
            continue

        paper_h = file_hash(paper_path)
        regen_h = file_hash(regen_path)
        paper_s = file_size(paper_path)
        regen_s = file_size(regen_path)

        if paper_h == regen_h:
            matches.append(fig_name)
        else:
            size_diff_pct = abs(regen_s - paper_s) / paper_s * 100
            # Check if the regenerated figure matches the pre-existing results/figures version
            # (which may be newer than paper/figures/)
            pre_hash = pre_regen_hashes.get(fig_name)
            regen_matches_previous = (pre_hash == regen_h) if pre_hash else False

            severity = "LOW" if size_diff_pct < 5 else "MEDIUM" if size_diff_pct < 20 else "HIGH"

            discrepancies.append(
                {
                    "figure": fig_name,
                    "issue": "content differs",
                    "severity": severity,
                    "paper_size_kb": round(paper_s / 1024, 1),
                    "regen_size_kb": round(regen_s / 1024, 1),
                    "size_diff_pct": round(size_diff_pct, 1),
                    "regen_is_deterministic": regen_matches_previous,
                }
            )

    # Report
    print("=" * 80)
    print("VERIFICATION RESULTS")
    print("=" * 80)

    print(f"\n  Exact matches:  {len(matches)} / {len(EXPECTED_FIGURES)}")
    print(f"  Discrepancies:  {len(discrepancies)}")

    if matches:
        print("\n  MATCHING figures (hash-identical to paper/figures/):")
        for f in matches:
            print(f"    [OK] {f}")

    if discrepancies:
        print("\n  DISCREPANCIES:")
        for d in discrepancies:
            issue = d["issue"]
            sev = d["severity"]
            fig = d["figure"]
            if "size_diff_pct" in d:
                detail = (
                    f"paper={d['paper_size_kb']}KB, regen={d['regen_size_kb']}KB "
                    f"({d['size_diff_pct']:+.1f}%), deterministic={d.get('regen_is_deterministic', 'N/A')}"
                )
                print(f"    [{sev}] {fig}: {issue} ({detail})")
            else:
                print(f"    [{sev}] {fig}: {issue}")

    # Interpretation
    print(f"\n{'=' * 80}")
    print("INTERPRETATION")
    print("=" * 80)

    if not discrepancies:
        print("\n  All figures are exactly reproducible. Paper figures match regenerated output.")
    elif all(d["severity"] == "LOW" for d in discrepancies if "severity" in d):
        print("\n  Minor discrepancies detected. These may be due to:")
        print("  - Matplotlib version differences")
        print("  - Font rendering differences")
        print("  - Floating point non-determinism in figure layout")
        print("  The paper figures are likely still valid.")
    else:
        stale_count = sum(1 for d in discrepancies if d.get("issue") == "content differs")
        if stale_count > 0:
            print(f"\n  {stale_count} figures in paper/figures/ appear STALE.")
            print("  The committed paper figures do not match the current data.")
            print("  Consider copying regenerated figures from results/figures/ to paper/figures/:")
            for d in discrepancies:
                if d.get("issue") == "content differs":
                    print(f"    cp results/figures/{d['figure']} paper/figures/{d['figure']}")

    print()
    return len(discrepancies)


if __name__ == "__main__":
    n_issues = main()
    sys.exit(0 if n_issues == 0 else 1)
