"""Extract per-task-order bandit posterior evolution data for web visualization.

Replays Beta-Bernoulli Thompson Sampling updates from episodes + feedback,
recording posterior snapshots at every 5th task_order for conditions C2, C3, C4.

Output: JSON array of {condition_id, task_order, preset_id, alpha, beta, mean}
"""

import json
import sqlite3
from collections import defaultdict
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "experiment_v3.db"
OUTPUT_PATH = Path.home() / "kusp-site" / "data" / "posterior_evolution.json"

# Checkpoints: 1, 5, 10, 15, ..., 300
CHECKPOINTS = {1} | set(range(5, 301, 5))

PRESET_IDS = [
    "balanced",
    "diversity_seeker",
    "exploration",
    "importance_heavy",
    "importance_relevance",
    "pure_importance",
    "pure_recency",
    "pure_relevance",
    "recency_heavy",
    "recency_importance",
    "recency_relevance",
    "relevance_heavy",
]

CONDITIONS = [2, 3, 4]


def compute_reward_c2_c3(
    rating_recency: int, rating_importance: int, rating_relevance: int
) -> float:
    """Likert 1-5 ratings -> [0, 1] reward. Same as normalize_likert_to_reward."""
    avg = (rating_recency + rating_importance + rating_relevance) / 3.0
    return (avg - 1.0) / 4.0


def compute_reward_c4(
    inferred_recency: float, inferred_importance: float, inferred_relevance: float
) -> float:
    """Inferred scores already in [0, 1] -> composite reward."""
    return (inferred_recency + inferred_importance + inferred_relevance) / 3.0


def main() -> None:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    # Load all episodes with feedback for C2-C4, ordered by task_order
    rows = conn.execute(
        """
        SELECT e.condition_id, e.task_order, e.preset_id,
               f.rating_recency, f.rating_importance, f.rating_relevance,
               f.inferred_recency, f.inferred_importance, f.inferred_relevance
        FROM episodes e
        LEFT JOIN feedback f ON e.episode_id = f.episode_id
        WHERE e.condition_id IN (2, 3, 4)
        ORDER BY e.condition_id, e.task_order
        """
    ).fetchall()

    # Initialize posteriors: {condition_id: {preset_id: [alpha, beta]}}
    posteriors: dict[int, dict[str, list[float]]] = {}
    for cid in CONDITIONS:
        posteriors[cid] = {pid: [1.0, 1.0] for pid in PRESET_IDS}

    # Group rows by condition for sequential replay
    by_condition: dict[int, list] = defaultdict(list)
    for row in rows:
        by_condition[row["condition_id"]].append(row)

    results = []

    # Snapshot at task_order=0 (prior state before any updates)
    # Not requested, but task_order=1 snapshot is taken AFTER the update at t=1

    for cid in CONDITIONS:
        # Reset posteriors for this condition
        state = {pid: [1.0, 1.0] for pid in PRESET_IDS}

        for row in by_condition[cid]:
            task_order = row["task_order"]
            preset_id = row["preset_id"]

            # Compute reward if feedback exists
            reward = None
            if cid in (2, 3):
                if row["rating_recency"] is not None:
                    reward = compute_reward_c2_c3(
                        row["rating_recency"],
                        row["rating_importance"],
                        row["rating_relevance"],
                    )
            elif cid == 4:
                if row["inferred_recency"] is not None:
                    reward = compute_reward_c4(
                        row["inferred_recency"],
                        row["inferred_importance"],
                        row["inferred_relevance"],
                    )

            # Update posterior if we got a reward
            if reward is not None:
                state[preset_id][0] += reward  # alpha += reward
                state[preset_id][1] += 1.0 - reward  # beta += (1 - reward)

            # Record snapshot at checkpoints
            if task_order in CHECKPOINTS:
                for pid in PRESET_IDS:
                    a, b = state[pid]
                    results.append(
                        {
                            "condition_id": cid,
                            "task_order": task_order,
                            "preset_id": pid,
                            "alpha": round(a, 6),
                            "beta": round(b, 6),
                            "mean": round(a / (a + b), 6),
                        }
                    )

    # Validate final state against bandit_state table
    final_state_db = conn.execute(
        """SELECT condition_id, preset_id, alpha, beta
           FROM bandit_state
           WHERE condition_id IN (2, 3, 4)
           ORDER BY condition_id, preset_id"""
    ).fetchall()

    # Rebuild final state from replay for validation
    final_replayed: dict[int, dict[str, list[float]]] = {
        cid: {pid: [1.0, 1.0] for pid in PRESET_IDS} for cid in CONDITIONS
    }
    for row in rows:
        cid = row["condition_id"]
        pid = row["preset_id"]
        reward = None
        if cid in (2, 3) and row["rating_recency"] is not None:
            reward = compute_reward_c2_c3(
                row["rating_recency"], row["rating_importance"], row["rating_relevance"]
            )
        elif cid == 4 and row["inferred_recency"] is not None:
            reward = compute_reward_c4(
                row["inferred_recency"],
                row["inferred_importance"],
                row["inferred_relevance"],
            )
        if reward is not None:
            final_replayed[cid][pid][0] += reward
            final_replayed[cid][pid][1] += 1.0 - reward

    mismatches = 0
    for db_row in final_state_db:
        cid = db_row["condition_id"]
        pid = db_row["preset_id"]
        db_alpha = db_row["alpha"]
        db_beta = db_row["beta"]
        rep_alpha, rep_beta = final_replayed[cid][pid]

        if abs(db_alpha - rep_alpha) > 0.01 or abs(db_beta - rep_beta) > 0.01:
            mismatches += 1
            print(
                f"MISMATCH C{cid} {pid}: "
                f"DB=({db_alpha:.4f}, {db_beta:.4f}) "
                f"Replay=({rep_alpha:.4f}, {rep_beta:.4f})"
            )

    if mismatches == 0:
        print("Validation passed: replay matches final bandit_state for all arms.")
    else:
        print(f"WARNING: {mismatches} mismatches found!")

    conn.close()

    # Write output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=None, separators=(",", ":"))

    n_records = len(results)
    n_checkpoints = len(CHECKPOINTS)
    file_size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(
        f"Wrote {n_records} records "
        f"({n_checkpoints} checkpoints x {len(CONDITIONS)} conditions x {len(PRESET_IDS)} presets)"
    )
    print(f"Output: {OUTPUT_PATH} ({file_size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
