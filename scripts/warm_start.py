#!/usr/bin/env python3
"""
Warm-start data extractor for bootstrapping compound experiment bandits.

Reads TIER 0 agent system data (episodic_memory.db) and converts historical
bandit learning into Beta distribution priors for the compound experiment.

Per ANALYSIS.md §6f: The 579 TIER 0 iterations are a prior that should not
be wasted. Initialize Beta priors from observed success rates per preset,
rather than uniform Beta(1,1). Expected impact: roughly halves convergence
time for the retrieval weight layer.

Sources:
  tier0  — Read from TIER 0 episodic_memory.db (weight_posteriors + engine_log)
  experiment — Fallback: read from the running experiment.db bandit_state

Environment variables:
  TIER0_DB  — Path to the TIER 0 episodic_memory.db file.
              Overrides the default search paths when set.

Usage:
  python scripts/warm_start.py                          # auto-detect source
  python scripts/warm_start.py --source tier0           # force TIER 0
  python scripts/warm_start.py --source experiment      # use experiment.db
  python scripts/warm_start.py --scale 0.3              # less weight on history
  python scripts/warm_start.py --dry-run                # print, don't write
  TIER0_DB=/path/to/episodic_memory.db python scripts/warm_start.py  # custom path
"""

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent

# ── Possible TIER 0 DB locations ──
# Set TIER0_DB environment variable to override the default search paths.
# Default paths search under $HOME, which only work on the original author's machine.


def _tier0_db_search_paths() -> list[Path]:
    """Return list of paths to search for the TIER 0 episodic_memory.db."""
    env_path = os.environ.get("TIER0_DB")
    if env_path:
        return [Path(env_path)]
    return [
        Path.home() / "agent-system" / "episodic_memory.db",
    ]


TIER0_DB_PATHS = _tier0_db_search_paths()

EXPERIMENT_DB_PATH = ROOT / "experiment_v3.db"
COMPOUND_DB_PATH = ROOT / "compound_experiment.db"
OUTPUT_PATH = ROOT / "data" / "warm_start_priors.json"

# ── TIER 0 → Experiment Preset Mapping ──
#
# TIER 0 weight presets operate on (relevance, success_rate, maturity) dimensions.
# The compound experiment operates on (recency, importance, relevance) dimensions.
# These are different dimensional spaces, so we cannot directly map presets 1:1.
#
# Instead, we extract the *aggregate* learning signal: what is the overall
# success rate of Thompson Sampling across all presets? This gives us an
# informative prior for the experiment's bandits — they should start knowing
# that "TS over weight presets works at roughly X% reward rate" rather than
# starting from a completely uninformative Beta(1,1).
#
# For presets with analogous names (balanced, relevance_heavy), we can also
# transfer the relative performance ranking.

TIER0_PRESETS = {
    "balanced": (0.50, 0.20, 0.30),  # (relevance, success, maturity)
    "relevance_heavy": (0.70, 0.15, 0.15),
    "success_heavy": (0.30, 0.50, 0.20),
    "maturity_heavy": (0.30, 0.15, 0.55),
    "relevance_success": (0.45, 0.40, 0.15),
}

EXPERIMENT_PRESETS = [
    "pure_relevance",
    "pure_recency",
    "pure_importance",
    "relevance_heavy",
    "recency_heavy",
    "importance_heavy",
    "recency_relevance",
    "importance_relevance",
    "recency_importance",
    "balanced",
    "diversity_seeker",
    "exploration",
]

# Approximate mapping: TIER 0 preset → experiment v3 preset
# Based on conceptual similarity of what each preset emphasizes
PRESET_MAP = {
    "balanced": "balanced",  # Both are balanced baselines
    "relevance_heavy": "relevance_heavy",  # Both emphasize relevance
    "success_heavy": "importance_heavy",  # Success ≈ importance (track record)
    "maturity_heavy": "recency_heavy",  # Maturity ≈ inverse of recency bias
    "relevance_success": "importance_relevance",  # Blends of similar concepts
}


def find_tier0_db() -> Path | None:
    """Locate the TIER 0 episodic_memory.db file."""
    for path in TIER0_DB_PATHS:
        if path.exists():
            return path
    return None


def connect_readonly(db_path: Path) -> sqlite3.Connection:
    """Open a read-only connection to a SQLite database."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    """Check if a table exists in the database."""
    row = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row[0] > 0


# ── TIER 0 Extraction ──


def extract_tier0_data(db_path: Path, scale: float) -> dict | None:
    """Extract warm-start priors from TIER 0 episodic_memory.db.

    Reads:
      1. weight_posteriors — per-agent Beta(alpha, beta) for each preset
      2. engine_log — per-iteration dimension ratings and preset usage

    Returns a warm-start dict or None if data is insufficient.
    """
    print(f"[tier0] Opening {db_path}")
    conn = connect_readonly(db_path)

    # Check required tables exist
    required = ["weight_posteriors"]
    optional = ["engine_log", "retrieval_feedback"]

    for t in required:
        if not table_exists(conn, t):
            print(f"[tier0] ERROR: Required table '{t}' not found.")
            conn.close()
            return None

    available_tables = [t for t in optional if table_exists(conn, t)]
    print(f"[tier0] Available tables: {required + available_tables}")

    # ── 1. Read weight posteriors (the core bandit state) ──

    rows = conn.execute("SELECT agent, preset_id, alpha, beta FROM weight_posteriors").fetchall()

    if not rows:
        print("[tier0] ERROR: weight_posteriors table is empty.")
        conn.close()
        return None

    print(f"[tier0] Found {len(rows)} weight_posteriors rows")

    # Aggregate across agents: for each preset, pool the evidence
    preset_totals: dict[str, dict] = {}
    agent_data: dict[str, list] = {}

    for row in rows:
        agent = row["agent"]
        preset_id = row["preset_id"]
        alpha = row["alpha"]
        beta = row["beta"]

        if preset_id not in preset_totals:
            preset_totals[preset_id] = {"alpha_sum": 0.0, "beta_sum": 0.0, "agents": 0}
        preset_totals[preset_id]["alpha_sum"] += alpha
        preset_totals[preset_id]["beta_sum"] += beta
        preset_totals[preset_id]["agents"] += 1

        if agent not in agent_data:
            agent_data[agent] = []
        agent_data[agent].append(
            {
                "preset_id": preset_id,
                "alpha": alpha,
                "beta": beta,
                "win_rate": alpha / (alpha + beta) if (alpha + beta) > 0 else 0.5,
            }
        )

    # ── 2. Read engine_log for iteration counts and dimension ratings ──

    iterations_extracted = 0
    dimension_stats = {}

    if "engine_log" in available_tables:
        # Count total iterations with skill selection (TIER 0 active iterations)
        count_row = conn.execute(
            "SELECT COUNT(*) FROM engine_log WHERE selected_skill_id IS NOT NULL"
        ).fetchone()
        iterations_extracted = count_row[0]
        print(f"[tier0] Total iterations with skill selection: {iterations_extracted}")

        # Dimension ratings (normalized 0-1 in TIER 0)
        dim_rows = conn.execute("""
            SELECT
                weight_preset_used,
                COUNT(*) as n,
                AVG(recency_rating) as avg_recency,
                AVG(importance_rating) as avg_importance,
                AVG(relevance_rating) as avg_relevance
            FROM engine_log
            WHERE recency_rating IS NOT NULL
              AND weight_preset_used IS NOT NULL
              AND weight_preset_used != ''
            GROUP BY weight_preset_used
        """).fetchall()

        for dr in dim_rows:
            preset = dr["weight_preset_used"]
            dimension_stats[preset] = {
                "n": dr["n"],
                "avg_recency": round(dr["avg_recency"], 4) if dr["avg_recency"] else None,
                "avg_importance": round(dr["avg_importance"], 4) if dr["avg_importance"] else None,
                "avg_relevance": round(dr["avg_relevance"], 4) if dr["avg_relevance"] else None,
            }

        if dimension_stats:
            print(f"[tier0] Preset effectiveness data for {len(dimension_stats)} presets:")
            for preset, stats in sorted(dimension_stats.items()):
                composite = 0.0
                count = 0
                for k in ("avg_recency", "avg_importance", "avg_relevance"):
                    if stats[k] is not None:
                        composite += stats[k]
                        count += 1
                avg = composite / count if count > 0 else 0.0
                print(
                    f"  {preset:25s}  n={stats['n']:3d}  "
                    f"R={stats['avg_recency']:.3f}  I={stats['avg_importance']:.3f}  "
                    f"V={stats['avg_relevance']:.3f}  avg={avg:.3f}"
                )

    # ── 3. Read retrieval_feedback for additional signal ──

    feedback_count = 0
    if "retrieval_feedback" in available_tables:
        count_row = conn.execute("SELECT COUNT(*) FROM retrieval_feedback").fetchone()
        feedback_count = count_row[0]
        print(f"[tier0] Retrieval feedback entries: {feedback_count}")

    conn.close()

    # ── 4. Compute warm-start priors ──

    # Strategy: For each TIER 0 preset, compute the average win_rate across
    # all agents. Map to the corresponding experiment preset. Scale the
    # evidence by the scale factor to control how much history influences
    # the new experiment.
    #
    # Formula: If TIER 0 aggregated posterior is Beta(alpha_pool, beta_pool),
    # the pooled success rate is s = alpha_pool / (alpha_pool + beta_pool).
    # Total pseudo-observations = alpha_pool + beta_pool - (2 * n_agents)
    #   (subtract the initial Beta(1,1) prior contribution from each agent)
    # Scaled observations = pseudo_obs * scale
    # New prior = Beta(1 + s * scaled_obs, 1 + (1-s) * scaled_obs)

    priors = {}
    tier0_success_rates = {}

    for t0_preset, totals in preset_totals.items():
        alpha_pool = totals["alpha_sum"]
        beta_pool = totals["beta_sum"]
        n_agents = totals["agents"]

        # Remove the uninformative prior contribution (Beta(1,1) per agent)
        observed_alpha = alpha_pool - n_agents  # successes
        observed_beta = beta_pool - n_agents  # failures
        pseudo_obs = max(0, observed_alpha + observed_beta)

        if pseudo_obs > 0:
            success_rate = observed_alpha / pseudo_obs
        else:
            success_rate = 0.5  # no real data, stay at 0.5

        tier0_success_rates[t0_preset] = {
            "success_rate": round(success_rate, 4),
            "pseudo_observations": round(pseudo_obs, 1),
            "pooled_alpha": round(alpha_pool, 2),
            "pooled_beta": round(beta_pool, 2),
            "n_agents": n_agents,
        }

        # Map to experiment preset
        exp_preset = PRESET_MAP.get(t0_preset)
        if exp_preset is None:
            print(f"  [warning] No mapping for TIER 0 preset '{t0_preset}', skipping")
            continue

        # Scale down the evidence
        scaled_obs = pseudo_obs * scale
        new_alpha = 1.0 + success_rate * scaled_obs
        new_beta = 1.0 + (1.0 - success_rate) * scaled_obs

        priors[exp_preset] = {
            "alpha": round(new_alpha, 4),
            "beta": round(new_beta, 4),
            "mean": round(new_alpha / (new_alpha + new_beta), 4),
            "source_preset": t0_preset,
            "source_success_rate": round(success_rate, 4),
            "scaled_observations": round(scaled_obs, 2),
        }

    # Fill any experiment presets that had no TIER 0 mapping with the
    # aggregate (global) prior from all TIER 0 data
    total_obs_alpha = sum(max(0, t["alpha_sum"] - t["agents"]) for t in preset_totals.values())
    total_obs_beta = sum(max(0, t["beta_sum"] - t["agents"]) for t in preset_totals.values())
    total_obs = total_obs_alpha + total_obs_beta

    if total_obs > 0:
        global_success_rate = total_obs_alpha / total_obs
    else:
        global_success_rate = 0.5

    for exp_preset in EXPERIMENT_PRESETS:
        if exp_preset not in priors:
            scaled_obs = total_obs * scale / len(EXPERIMENT_PRESETS)
            new_alpha = 1.0 + global_success_rate * scaled_obs
            new_beta = 1.0 + (1.0 - global_success_rate) * scaled_obs
            priors[exp_preset] = {
                "alpha": round(new_alpha, 4),
                "beta": round(new_beta, 4),
                "mean": round(new_alpha / (new_alpha + new_beta), 4),
                "source_preset": "global_aggregate",
                "source_success_rate": round(global_success_rate, 4),
                "scaled_observations": round(scaled_obs, 2),
            }

    # ── 5. Build output ──

    result = {
        "source": "tier0_episodic_memory",
        "source_path": str(db_path),
        "iterations_extracted": iterations_extracted,
        "feedback_entries": feedback_count,
        "scale_factor": scale,
        "agents_found": sorted(agent_data.keys()),
        "tier0_presets_found": sorted(preset_totals.keys()),
        "preset_mapping": PRESET_MAP,
        "priors": priors,
        "tier0_success_rates": tier0_success_rates,
        "dimension_stats": dimension_stats,
        "global_success_rate": round(global_success_rate, 4),
        "total_pseudo_observations": round(total_obs, 1),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    return result


# ── Experiment DB Fallback Extraction ──


def extract_experiment_data(db_path: Path, scale: float) -> dict | None:
    """Fallback: extract warm-start priors from the running experiment.db.

    Uses the first 40 tasks worth of bandit state to bootstrap the compound
    experiment. Scaled down by the scale factor to avoid over-fitting to
    early noisy data.
    """
    if not db_path.exists():
        print(f"[experiment] ERROR: {db_path} not found.")
        return None

    print(f"[experiment] Opening {db_path}")
    conn = connect_readonly(db_path)

    if not table_exists(conn, "bandit_state"):
        print("[experiment] ERROR: bandit_state table not found.")
        conn.close()
        return None

    # Read bandit state
    rows = conn.execute(
        "SELECT condition_id, preset_id, alpha, beta, pulls, total_reward FROM bandit_state"
    ).fetchall()

    if not rows:
        print("[experiment] ERROR: bandit_state table is empty.")
        conn.close()
        return None

    # Count total episodes
    episode_count = 0
    if table_exists(conn, "episodes"):
        count_row = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()
        episode_count = count_row[0]

    print(f"[experiment] Found {len(rows)} bandit_state rows, {episode_count} episodes")

    # Aggregate across conditions: for each preset, pool the evidence
    preset_totals: dict[str, dict] = {}
    for row in rows:
        preset_id = row["preset_id"]
        alpha = row["alpha"]
        beta = row["beta"]
        pulls = row["pulls"]

        if preset_id not in preset_totals:
            preset_totals[preset_id] = {
                "alpha_sum": 0.0,
                "beta_sum": 0.0,
                "total_pulls": 0,
                "conditions": 0,
            }
        preset_totals[preset_id]["alpha_sum"] += alpha
        preset_totals[preset_id]["beta_sum"] += beta
        preset_totals[preset_id]["total_pulls"] += pulls
        preset_totals[preset_id]["conditions"] += 1

    # Cap at first 40 tasks worth of data to avoid over-fitting
    # If total pulls > 40, scale proportionally
    total_pulls = sum(t["total_pulls"] for t in preset_totals.values())
    cap = 40
    pull_scale = min(1.0, cap / total_pulls) if total_pulls > 0 else 1.0

    priors = {}
    skill_success_rates = {}

    for preset_id, totals in preset_totals.items():
        n_conditions = totals["conditions"]
        # Remove prior contribution
        observed_alpha = totals["alpha_sum"] - n_conditions
        observed_beta = totals["beta_sum"] - n_conditions
        pseudo_obs = max(0, observed_alpha + observed_beta)

        if pseudo_obs > 0:
            success_rate = observed_alpha / pseudo_obs
        else:
            success_rate = 0.5

        skill_success_rates[preset_id] = {
            "success_rate": round(success_rate, 4),
            "pseudo_observations": round(pseudo_obs, 1),
            "total_pulls": totals["total_pulls"],
        }

        # Apply both the 40-task cap and the user's scale factor
        effective_scale = scale * pull_scale
        scaled_obs = pseudo_obs * effective_scale
        new_alpha = 1.0 + success_rate * scaled_obs
        new_beta = 1.0 + (1.0 - success_rate) * scaled_obs

        priors[preset_id] = {
            "alpha": round(new_alpha, 4),
            "beta": round(new_beta, 4),
            "mean": round(new_alpha / (new_alpha + new_beta), 4),
            "source_preset": preset_id,
            "source_success_rate": round(success_rate, 4),
            "scaled_observations": round(scaled_obs, 2),
        }

    conn.close()

    # Ensure all experiment presets are covered
    for exp_preset in EXPERIMENT_PRESETS:
        if exp_preset not in priors:
            priors[exp_preset] = {
                "alpha": 1.0,
                "beta": 1.0,
                "mean": 0.5,
                "source_preset": "uninformative",
                "source_success_rate": 0.5,
                "scaled_observations": 0.0,
            }

    result = {
        "source": "experiment_db",
        "source_path": str(db_path),
        "iterations_extracted": episode_count,
        "feedback_entries": 0,
        "scale_factor": scale,
        "pull_cap": cap,
        "pull_scale": round(pull_scale, 4),
        "agents_found": [],
        "priors": priors,
        "skill_success_rates": skill_success_rates,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    return result


# ── Main ──


def print_summary(result: dict) -> None:
    """Print a human-readable summary of the warm-start extraction."""
    print("\n" + "=" * 70)
    print("WARM-START EXTRACTION SUMMARY")
    print("=" * 70)
    print(f"  Source:              {result['source']}")
    print(f"  Source path:         {result['source_path']}")
    print(f"  Iterations:          {result['iterations_extracted']}")
    print(f"  Scale factor:        {result['scale_factor']}")
    if result.get("agents_found"):
        print(f"  Agents:              {', '.join(result['agents_found'])}")
    if result.get("global_success_rate") is not None:
        print(f"  Global success rate: {result['global_success_rate']:.4f}")
    if result.get("total_pseudo_observations") is not None:
        print(f"  Total observations:  {result['total_pseudo_observations']:.0f}")

    print(f"\n  Generated priors ({len(result['priors'])} presets):")
    print(f"  {'Preset':25s} {'Alpha':>8s} {'Beta':>8s} {'Mean':>8s} {'Source':>25s}")
    print(f"  {'-' * 25} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 25}")
    for preset_id, prior in sorted(result["priors"].items()):
        print(
            f"  {preset_id:25s} "
            f"{prior['alpha']:8.4f} "
            f"{prior['beta']:8.4f} "
            f"{prior['mean']:8.4f} "
            f"{prior.get('source_preset', ''):>25s}"
        )

    # Show how much the priors differ from uninformative Beta(1,1)
    print("\n  Prior informativeness (distance from Beta(1,1) mean=0.5):")
    for preset_id, prior in sorted(result["priors"].items()):
        delta = prior["mean"] - 0.5
        bar_len = int(abs(delta) * 100)
        direction = "+" if delta >= 0 else "-"
        bar = direction * bar_len if bar_len > 0 else "="
        print(f"  {preset_id:25s} {delta:+.4f}  {bar}")

    print(f"\n  Generated at: {result['generated_at']}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract warm-start priors from TIER 0 data for compound experiment bandits.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source",
        choices=["tier0", "experiment", "auto"],
        default="auto",
        help="Data source: 'tier0' (episodic_memory.db), 'experiment' (experiment.db), "
        "or 'auto' (try tier0 first, fall back to experiment). Default: auto",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.5,
        help="Scale factor for historical data (0.0-1.0). Controls how much weight "
        "to give to historical observations. Default: 0.5 (half weight).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_PATH),
        help=f"Output path for warm-start JSON. Default: {OUTPUT_PATH}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary but don't write the output file.",
    )
    args = parser.parse_args()

    if not (0.0 < args.scale <= 1.0):
        print(f"ERROR: --scale must be in (0.0, 1.0], got {args.scale}")
        sys.exit(1)

    result = None

    # ── Try TIER 0 source ──
    if args.source in ("tier0", "auto"):
        tier0_path = find_tier0_db()
        if tier0_path:
            print(f"[main] Found TIER 0 DB at {tier0_path}")
            result = extract_tier0_data(tier0_path, args.scale)
            if result is None and args.source == "tier0":
                print("[main] TIER 0 extraction failed and --source=tier0 was specified.")
                sys.exit(1)
        else:
            msg = "[main] TIER 0 episodic_memory.db not found. Searched:\n" + "\n".join(
                f"  - {p}" for p in TIER0_DB_PATHS
            )
            if args.source == "tier0":
                print(msg)
                print("[main] Cannot continue with --source=tier0.")
                sys.exit(1)
            else:
                print(msg)
                print("[main] Falling back to experiment DB...")

    # ── Try experiment DB source ──
    if result is None and args.source in ("experiment", "auto"):
        # Try compound_experiment.db first, then experiment.db
        for db_path in [COMPOUND_DB_PATH, EXPERIMENT_DB_PATH]:
            if db_path.exists():
                result = extract_experiment_data(db_path, args.scale)
                if result is not None:
                    break

    if result is None:
        print("\n[main] No data source available for warm-starting.")
        print("  Possible causes:")
        print("  - TIER 0 episodic_memory.db not found at expected paths")
        print("  - experiment.db has no bandit_state data")
        print("")
        print("  The compound experiment will start from uninformative Beta(1,1) priors.")
        print("  This is fine — it just means ~2x more iterations to converge.")
        sys.exit(0)

    # ── Print summary ──
    print_summary(result)

    # ── Write output ──
    if args.dry_run:
        print(f"\n[dry-run] Would write to {args.output}")
        print("[dry-run] JSON preview (first 500 chars):")
        preview = json.dumps(result, indent=2)[:500]
        print(preview)
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n[main] Wrote warm-start priors to {output_path}")
        print("[main] Load in compound experiment with:")
        print(f"  with open('{output_path}') as f:")
        print("      priors = json.load(f)")
        print("  init_bandit_state(conn, conditions, presets,")
        print("      prior_alpha=priors['priors']['balanced']['alpha'],")
        print("      prior_beta=priors['priors']['balanced']['beta'])")


if __name__ == "__main__":
    main()
