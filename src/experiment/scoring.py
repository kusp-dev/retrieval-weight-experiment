"""
Dynamic skill scoring for Stage 2 retrieval weight application.

Computes per-skill, per-condition recency and importance scores that evolve
during the experiment. Each condition builds its own usage history independently.

Design ref: SYSTEM_DESIGN.md §6, §13 (resolved decisions)

Formulas:
  - recency:    max(0, 1 - iterations_since_last_use / decay_window)
                Skills never used start at 0.5.
  - importance: (success_count + 1) / (success_count + failure_count + 2)
                Laplace-smoothed success rate. Starts at 0.5 (uniform prior).
  - relevance:  cosine_similarity(skill_embedding, task_embedding)
                Computed by the search engine, not here.
"""

import sqlite3


def compute_recency(
    conn: sqlite3.Connection,
    condition_id: int,
    skill_id: str,
    current_task_order: int,
    decay_window: int = 50,
) -> float:
    """
    Compute recency score for a skill in a given condition.

    recency = max(0, 1 - iterations_since_last_use / decay_window)
    Skills never used in this condition start at 0.5.
    """
    row = conn.execute(
        """SELECT MAX(used_at_order) as last_used
           FROM skill_usage
           WHERE condition_id = ? AND skill_id = ?""",
        (condition_id, skill_id),
    ).fetchone()

    if row is None or row["last_used"] is None:
        return 0.5  # never used → neutral

    iterations_since = current_task_order - row["last_used"]
    return max(0.0, 1.0 - iterations_since / decay_window)


def compute_importance(
    conn: sqlite3.Connection,
    condition_id: int,
    skill_id: str,
) -> float:
    """
    Compute importance score for a skill in a given condition.

    importance = (success_count + 1) / (success_count + failure_count + 2)
    Laplace smoothing: starts at 0.5 (uniform Beta prior), updates with each use.
    """
    row = conn.execute(
        """SELECT
               COALESCE(SUM(CASE WHEN task_success = 1 THEN 1 ELSE 0 END), 0) as successes,
               COALESCE(SUM(CASE WHEN task_success = 0 THEN 1 ELSE 0 END), 0) as failures
           FROM skill_usage
           WHERE condition_id = ? AND skill_id = ?""",
        (condition_id, skill_id),
    ).fetchone()

    successes = row["successes"]
    failures = row["failures"]
    return (successes + 1) / (successes + failures + 2)


def record_skill_usage(
    conn: sqlite3.Connection,
    condition_id: int,
    skill_id: str,
    episode_id: int,
    task_success: int | None,
    task_order: int,
) -> None:
    """Record that a skill was used in an episode, for recency/importance tracking."""
    conn.execute(
        """INSERT OR REPLACE INTO skill_usage
           (condition_id, skill_id, episode_id, task_success, used_at_order)
           VALUES (?, ?, ?, ?, ?)""",
        (condition_id, skill_id, episode_id, task_success, task_order),
    )


def score_search_results(
    conn: sqlite3.Connection,
    condition_id: int,
    current_task_order: int,
    search_results: list,
    decay_window: int = 50,
) -> None:
    """
    Populate recency_score and importance_score on SearchResult objects.

    relevance_score is already set by the search engine (= dense cosine similarity).
    This function fills in the other two dimensions so that Stage 2 weight application
    produces meaningful scores.
    """
    for result in search_results:
        result.recency_score = compute_recency(
            conn, condition_id, result.skill_id, current_task_order, decay_window
        )
        result.importance_score = compute_importance(conn, condition_id, result.skill_id)
        # relevance_score is the dense cosine similarity from the search engine
        result.relevance_score = result.dense_score
