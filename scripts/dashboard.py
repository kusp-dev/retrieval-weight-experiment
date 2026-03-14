#!/usr/bin/env python3
"""
Experiment dashboard — comprehensive metrics for the retrieval weight paper.

Navigation: Overview | Theme 1 | ... | Theme 5
Within each theme: 4 conditions with full metric breakdown.
Supports multiple experiment runs via DB file selector.

Metrics tracked (SYSTEM_DESIGN.md §12):
  Primary: Task success, cumulative regret, NDCG@k, MRR, step efficiency, token use
  Secondary: Dimension trajectories (R/I/V), bandit convergence, reward trends
  Diagnostic: Parse rates, reward drift, C3 embedding differentiator
  Novel: Exploration ratio, preset entropy, cross-theme transfer

Usage:
  .venv313/bin/python scripts/dashboard.py
"""

import argparse
import json
import math
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path

from flask import Flask, Response, jsonify, request

ROOT = Path(__file__).parent.parent
app = Flask(__name__)

COND_NAMES = {1: "Control", 2: "Dim Feedback", 3: "Full System", 4: "Qualitative"}
COND_SHORT = {1: "C1 No FB", 2: "C2 Likert", 3: "C3 Full", 4: "C4 Qual"}
COND_COLORS = {1: "#9ca3af", 2: "#60a5fa", 3: "#34d399", 4: "#a78bfa"}
TOTAL_TASKS = 300
TOTAL_EPISODES = 1200


def list_runs():
    runs = []
    for p in sorted(ROOT.glob("experiment*.db")):
        if p.name.endswith("-wal") or p.name.endswith("-shm"):
            continue
        label = p.stem.replace("experiment_", "Run: ").replace("experiment", "Current Run")
        runs.append({"file": p.name, "label": label, "path": str(p)})
    return runs


def get_db(db_file=None):
    path = ROOT / (db_file or "experiment_v3.db")
    if not path.exists():
        return None
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def get_themes(conn):
    return [
        r["theme"]
        for r in conn.execute("SELECT DISTINCT theme FROM tasks ORDER BY theme").fetchall()
    ]


# ── Metric computations ──


def compute_ndcg_mrr(conn, condition_id=None, task_ids=None):
    """Compute NDCG@5 and MRR from retrieval_results vs ground_truth_skills."""
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

    ndcg_scores = []
    rr_scores = []

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
        dcg = 0.0
        idcg = 0.0
        k = min(5, len(retrieved))
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
        "ndcg5": round(sum(ndcg_scores) / len(ndcg_scores), 3) if ndcg_scores else None,
        "mrr": round(sum(rr_scores) / len(rr_scores), 3) if rr_scores else None,
        "count": len(ndcg_scores),
    }


def compute_cumulative_regret(conn, condition_id):
    """Cumulative regret: sum of (1.0 - reward) over episodes with parsed feedback."""
    rows = conn.execute(
        """
        SELECT COALESCE(
            (f.rating_recency + f.rating_importance + f.rating_relevance - 3) / 12.0,
            (f.inferred_recency + f.inferred_importance + f.inferred_relevance) / 3.0
        ) as reward
        FROM feedback f
        JOIN episodes e ON f.episode_id = e.episode_id
        WHERE e.condition_id = ?
        ORDER BY e.task_order
    """,
        (condition_id,),
    ).fetchall()
    if not rows:
        return {"regret": None, "rewards": []}

    regret = 0.0
    points = []
    for i, r in enumerate(rows):
        regret += 1.0 - (r["reward"] or 0)
        if (i + 1) % 5 == 0 or i == len(rows) - 1:
            points.append({"iter": i + 1, "regret": round(regret, 2)})
    return {"regret": round(regret, 2), "points": points}


def compute_reward_trend(conn, condition_id, window=10, task_ids=None):
    """Rolling average of composite reward over iterations."""
    query = """
        SELECT e.task_order,
            COALESCE(
                (f.rating_recency + f.rating_importance + f.rating_relevance - 3) / 12.0,
                (f.inferred_recency + f.inferred_importance + f.inferred_relevance) / 3.0
            ) as reward
        FROM feedback f
        JOIN episodes e ON f.episode_id = e.episode_id
        WHERE e.condition_id = ?
    """
    params = [condition_id]
    if task_ids:
        ph = ",".join("?" * len(task_ids))
        query += f" AND e.task_id IN ({ph})"
        params.extend(task_ids)
    query += " ORDER BY e.task_order"
    rows = conn.execute(query, params).fetchall()
    if not rows:
        return []
    rewards = [r["reward"] for r in rows]
    points = []
    for i in range(len(rewards)):
        start = max(0, i - window + 1)
        avg = sum(rewards[start : i + 1]) / (i - start + 1)
        points.append({"iter": rows[i]["task_order"], "value": round(avg, 3)})
    return points


def compute_dimension_trends(conn, condition_id, window=10, task_ids=None):
    """Rolling averages of R/I/V dimensions over iterations."""
    query = """
        SELECT e.task_order,
            COALESCE((f.rating_recency - 1) / 4.0, f.inferred_recency) as rec,
            COALESCE((f.rating_importance - 1) / 4.0, f.inferred_importance) as imp,
            COALESCE((f.rating_relevance - 1) / 4.0, f.inferred_relevance) as rel
        FROM feedback f
        JOIN episodes e ON f.episode_id = e.episode_id
        WHERE e.condition_id = ?
    """
    params = [condition_id]
    if task_ids:
        ph = ",".join("?" * len(task_ids))
        query += f" AND e.task_id IN ({ph})"
        params.extend(task_ids)
    query += " ORDER BY e.task_order"
    rows = conn.execute(query, params).fetchall()
    if not rows:
        return {"recency": [], "importance": [], "relevance": []}

    def rolling(vals):
        pts = []
        for i in range(len(vals)):
            start = max(0, i - window + 1)
            chunk = [v for v in vals[start : i + 1] if v is not None]
            avg = sum(chunk) / len(chunk) if chunk else 0
            pts.append(round(avg, 3))
        return pts

    orders = [r["task_order"] for r in rows]
    return {
        "orders": orders,
        "recency": rolling([r["rec"] for r in rows]),
        "importance": rolling([r["imp"] for r in rows]),
        "relevance": rolling([r["rel"] for r in rows]),
    }


def compute_token_trend(conn, condition_id, window=10, task_ids=None):
    """Rolling average token usage per episode over iterations."""
    query = "SELECT task_order, total_tokens FROM episodes WHERE condition_id = ?"
    params = [condition_id]
    if task_ids:
        ph = ",".join("?" * len(task_ids))
        query += f" AND task_id IN ({ph})"
        params.extend(task_ids)
    query += " ORDER BY task_order"
    rows = conn.execute(query, params).fetchall()
    if not rows:
        return []
    points = []
    tokens = [r["total_tokens"] for r in rows]
    for i in range(len(tokens)):
        start = max(0, i - window + 1)
        avg = sum(tokens[start : i + 1]) / (i - start + 1)
        points.append({"iter": rows[i]["task_order"], "value": round(avg, 0)})
    return points


def compute_input_token_trend(conn, condition_id, window=10, task_ids=None):
    """Rolling average input token usage per episode over iterations."""
    query = (
        "SELECT task_order, input_tokens FROM episodes"
        " WHERE condition_id = ? AND input_tokens IS NOT NULL"
    )
    params = [condition_id]
    if task_ids:
        ph = ",".join("?" * len(task_ids))
        query += f" AND task_id IN ({ph})"
        params.extend(task_ids)
    query += " ORDER BY task_order"
    rows = conn.execute(query, params).fetchall()
    if not rows:
        return []
    points = []
    tokens = [r["input_tokens"] for r in rows]
    for i in range(len(tokens)):
        start = max(0, i - window + 1)
        avg = sum(tokens[start : i + 1]) / (i - start + 1)
        points.append({"iter": rows[i]["task_order"], "value": round(avg, 0)})
    return points


def compute_output_token_trend(conn, condition_id, window=10, task_ids=None):
    """Rolling average output token usage per episode over iterations."""
    query = (
        "SELECT task_order, output_tokens FROM episodes"
        " WHERE condition_id = ? AND output_tokens IS NOT NULL"
    )
    params = [condition_id]
    if task_ids:
        ph = ",".join("?" * len(task_ids))
        query += f" AND task_id IN ({ph})"
        params.extend(task_ids)
    query += " ORDER BY task_order"
    rows = conn.execute(query, params).fetchall()
    if not rows:
        return []
    points = []
    tokens = [r["output_tokens"] for r in rows]
    for i in range(len(tokens)):
        start = max(0, i - window + 1)
        avg = sum(tokens[start : i + 1]) / (i - start + 1)
        points.append({"iter": rows[i]["task_order"], "value": round(avg, 0)})
    return points


def compute_step_trend(conn, condition_id, window=10, task_ids=None):
    """Rolling average step count over iterations."""
    query = "SELECT task_order, step_count FROM episodes WHERE condition_id = ?"
    params = [condition_id]
    if task_ids:
        ph = ",".join("?" * len(task_ids))
        query += f" AND task_id IN ({ph})"
        params.extend(task_ids)
    query += " ORDER BY task_order"
    rows = conn.execute(query, params).fetchall()
    if not rows:
        return []
    points = []
    steps = [r["step_count"] for r in rows]
    for i in range(len(steps)):
        start = max(0, i - window + 1)
        avg = sum(steps[start : i + 1]) / (i - start + 1)
        points.append({"iter": rows[i]["task_order"], "value": round(avg, 2)})
    return points


def compute_exploration_ratio(conn, condition_id):
    """Fraction of pulls on non-best arm (exploration indicator)."""
    arms = conn.execute(
        "SELECT preset_id, pulls FROM bandit_state WHERE condition_id = ? ORDER BY pulls DESC",
        (condition_id,),
    ).fetchall()
    if not arms:
        return None
    total = sum(a["pulls"] for a in arms)
    if total == 0:
        return None
    best_pulls = arms[0]["pulls"]
    return round((total - best_pulls) / total, 3)


def compute_preset_entropy(conn, condition_id):
    """Shannon entropy of preset selections (higher = more exploration)."""
    arms = conn.execute(
        "SELECT pulls FROM bandit_state WHERE condition_id = ?", (condition_id,)
    ).fetchall()
    total = sum(a["pulls"] for a in arms)
    if total == 0:
        return None
    entropy = 0.0
    for a in arms:
        if a["pulls"] > 0:
            p = a["pulls"] / total
            entropy -= p * math.log2(p)
    max_entropy = math.log2(len(arms)) if len(arms) > 0 else 1
    return round(entropy / max_entropy, 3)  # Normalized 0-1


def compute_bandit_history(conn, condition_id):
    """Build the posterior mean history for the best arm over iterations.

    Walk through episodes in order, replaying the bandit updates to
    reconstruct how the best arm's posterior mean evolved.
    """
    # Get all episodes with their preset and reward for this condition
    rows = conn.execute(
        """
        SELECT e.task_order, e.preset_id,
            COALESCE(
                (f.rating_recency + f.rating_importance + f.rating_relevance - 3) / 12.0,
                (f.inferred_recency + f.inferred_importance + f.inferred_relevance) / 3.0
            ) as reward
        FROM episodes e
        LEFT JOIN feedback f ON e.episode_id = f.episode_id
        WHERE e.condition_id = ? AND e.preset_id IS NOT NULL
        ORDER BY e.task_order
    """,
        (condition_id,),
    ).fetchall()

    if not rows:
        return {"best_arm_history": [], "arm_histories": {}}

    # Get all preset IDs for this condition
    presets = [
        r["preset_id"]
        for r in conn.execute(
            "SELECT DISTINCT preset_id FROM bandit_state WHERE condition_id = ?", (condition_id,)
        ).fetchall()
    ]

    # Replay: start with uniform priors
    alphas = {p: 1.0 for p in presets}
    betas = {p: 1.0 for p in presets}

    best_arm_history = []
    arm_histories = {p: [] for p in presets}

    for row in rows:
        preset = row["preset_id"]
        reward = row["reward"]

        # Update posterior if we have feedback
        if reward is not None and preset in alphas:
            alphas[preset] += reward
            betas[preset] += 1.0 - reward

        # Compute current means
        means = {p: alphas[p] / (alphas[p] + betas[p]) for p in presets}
        best_preset = max(means, key=means.get)
        best_mean = means[best_preset]

        best_arm_history.append(
            {
                "iter": row["task_order"],
                "mean": round(best_mean, 4),
                "arm": best_preset,
            }
        )

        for p in presets:
            arm_histories[p].append(
                {
                    "iter": row["task_order"],
                    "mean": round(means[p], 4),
                }
            )

    return {"best_arm_history": best_arm_history, "arm_histories": arm_histories}


def compute_parse_rate_trend(conn, window=10):
    """Rolling parse rate over episodes (across conditions 2-4).

    Returns list of {iter, rate} where rate is fraction of last `window`
    episodes (conditions 2-4) that had parsed feedback.
    """
    rows = conn.execute("""
        SELECT e.episode_id, e.task_order, e.condition_id,
               CASE WHEN f.feedback_id IS NOT NULL THEN 1 ELSE 0 END as has_fb
        FROM episodes e
        LEFT JOIN feedback f ON e.episode_id = f.episode_id
        WHERE e.condition_id IN (2, 3, 4)
        ORDER BY e.episode_id
    """).fetchall()

    if not rows:
        return []

    points = []
    parsed_flags = []
    for row in rows:
        parsed_flags.append(row["has_fb"])
        start = max(0, len(parsed_flags) - window)
        chunk = parsed_flags[start:]
        rate = sum(chunk) / len(chunk)
        points.append({"iter": len(parsed_flags), "rate": round(rate * 100, 1)})

    return points


def compute_completion_rate(conn):
    """Compute tasks per hour over the last 10 tasks and projected completion."""
    # Get completion timestamps for all episodes, ordered
    rows = conn.execute("""
        SELECT completed_at FROM episodes
        WHERE completed_at IS NOT NULL
        ORDER BY episode_id
    """).fetchall()

    if len(rows) < 2:
        return {"recent_rate": None, "projected_completion": None, "rate_points": []}

    # Parse timestamps
    timestamps = []
    for r in rows:
        try:
            ts = datetime.strptime(r["completed_at"], "%Y-%m-%d %H:%M:%S")
            timestamps.append(ts)
        except (ValueError, TypeError):
            continue

    if len(timestamps) < 2:
        return {"recent_rate": None, "projected_completion": None, "rate_points": []}

    # Compute rolling rate (episodes per hour) over sliding windows of 10
    rate_points = []
    window = 10
    for i in range(window, len(timestamps)):
        dt = (timestamps[i] - timestamps[i - window]).total_seconds()
        if dt > 0:
            rate = window / dt * 3600  # episodes per hour
            rate_points.append({"iter": i + 1, "rate": round(rate, 2)})

    # Recent rate: last 10 tasks
    recent_window = min(10, len(timestamps) - 1)
    dt_recent = (timestamps[-1] - timestamps[-1 - recent_window]).total_seconds()
    recent_rate = recent_window / dt_recent * 3600 if dt_recent > 0 else 0

    # Projected completion: how many episodes remain, at recent rate
    total_episodes_done = len(timestamps)
    remaining = max(0, TOTAL_EPISODES - total_episodes_done)
    projected_hours = remaining / recent_rate if recent_rate > 0 else None
    projected_completion = None
    if projected_hours is not None:
        projected_completion = (datetime.now() + timedelta(hours=projected_hours)).strftime(
            "%b %d %H:%M"
        )

    return {
        "recent_rate": round(recent_rate, 1),
        "projected_completion": projected_completion,
        "projected_hours": round(projected_hours, 1) if projected_hours else None,
        "rate_points": rate_points,
    }


def compute_reward_drift(conn, condition_id, window=25):
    """Check for reward drift (upward trend = possible reward hacking)."""
    rows = conn.execute(
        """
        SELECT COALESCE(
            (f.rating_recency + f.rating_importance + f.rating_relevance - 3) / 12.0,
            (f.inferred_recency + f.inferred_importance + f.inferred_relevance) / 3.0
        ) as reward
        FROM feedback f
        JOIN episodes e ON f.episode_id = e.episode_id
        WHERE e.condition_id = ?
        ORDER BY e.task_order
    """,
        (condition_id,),
    ).fetchall()

    if len(rows) < window * 2:
        return {"status": "insufficient_data", "first_half": None, "second_half": None}

    rewards = [r["reward"] for r in rows]
    mid = len(rewards) // 2
    first_avg = sum(rewards[:mid]) / mid
    second_avg = sum(rewards[mid:]) / (len(rewards) - mid)
    drift = second_avg - first_avg

    status = "ok" if abs(drift) < 0.05 else ("warning" if drift > 0 else "negative_drift")
    return {
        "status": status,
        "first_half": round(first_avg, 3),
        "second_half": round(second_avg, 3),
        "drift": round(drift, 3),
    }


# ── API data builders ──


def _condition_stats(conn, task_ids=None):
    conditions = []
    for cid in range(1, 5):
        if task_ids:
            ph = ",".join("?" * len(task_ids))
            params = task_ids + [cid]
            where = f"task_id IN ({ph}) AND condition_id = ?"
            fb_where = f"e.task_id IN ({ph}) AND e.condition_id = ?"
        else:
            params = [cid]
            where = "condition_id = ?"
            fb_where = "e.condition_id = ?"

        row = conn.execute(
            f"SELECT COUNT(*) as n, AVG(total_tokens) as avg_tok,"
            f" AVG(step_count) as avg_steps,"
            f" SUM(total_tokens) as sum_tok,"
            f" AVG(input_tokens) as avg_in_tok,"
            f" AVG(output_tokens) as avg_out_tok"
            f" FROM episodes WHERE {where}",
            params,
        ).fetchone()
        fb_count = conn.execute(
            f"SELECT COUNT(*) FROM feedback f JOIN episodes e ON f.episode_id = e.episode_id "
            f"WHERE {fb_where}",
            params,
        ).fetchone()[0]
        avg_r = conn.execute(
            f"""SELECT AVG(COALESCE(
                    (rating_recency + rating_importance + rating_relevance - 3) / 12.0,
                    (inferred_recency + inferred_importance + inferred_relevance) / 3.0
                )) as v FROM feedback f
                JOIN episodes e ON f.episode_id = e.episode_id WHERE {fb_where}""",
            params,
        ).fetchone()["v"]
        dim_row = conn.execute(
            f"""SELECT
                 AVG(COALESCE((rating_recency-1)/4.0, inferred_recency)) as r,
                 AVG(COALESCE((rating_importance-1)/4.0, inferred_importance)) as i,
                 AVG(COALESCE((rating_relevance-1)/4.0, inferred_relevance)) as v
                FROM feedback f JOIN episodes e ON f.episode_id = e.episode_id WHERE {fb_where}""",
            params,
        ).fetchone()

        n = row["n"] or 0

        # Retrieval quality
        ir = compute_ndcg_mrr(conn, cid, task_ids)

        c = {
            "id": cid,
            "name": COND_NAMES[cid],
            "short": COND_SHORT[cid],
            "color": COND_COLORS[cid],
            "episodes": n,
            "parse_rate": round(fb_count / n * 100) if n > 0 else 0,
            "avg_reward": round(avg_r, 3) if avg_r else None,
            "avg_tokens": round(row["avg_tok"]) if row["avg_tok"] else 0,
            "avg_input_tokens": round(row["avg_in_tok"]) if row["avg_in_tok"] else 0,
            "avg_output_tokens": round(row["avg_out_tok"]) if row["avg_out_tok"] else 0,
            "avg_steps": round(row["avg_steps"], 1) if row["avg_steps"] else 0,
            "total_tokens": row["sum_tok"] or 0,
            "avg_recency": round(dim_row["r"], 3) if dim_row["r"] else None,
            "avg_importance": round(dim_row["i"], 3) if dim_row["i"] else None,
            "avg_relevance": round(dim_row["v"], 3) if dim_row["v"] else None,
            "ndcg5": ir["ndcg5"],
            "mrr": ir["mrr"],
        }
        conditions.append(c)
    return conditions


def _bandit_state(conn):
    bandits = []
    for cid in [2, 3, 4]:
        arms = conn.execute(
            "SELECT preset_id, alpha, beta, pulls, total_reward "
            "FROM bandit_state WHERE condition_id = ? "
            "ORDER BY (alpha / (alpha + beta)) DESC",
            (cid,),
        ).fetchall()
        arm_data = [
            {
                "preset": a["preset_id"],
                "pulls": a["pulls"],
                "mean": (
                    round(a["alpha"] / (a["alpha"] + a["beta"]), 3)
                    if (a["alpha"] + a["beta"]) > 0
                    else 0.5
                ),
                "alpha": round(a["alpha"], 2),
                "beta": round(a["beta"], 2),
            }
            for a in arms
        ]

        regret = compute_cumulative_regret(conn, cid)
        bandits.append(
            {
                "condition_id": cid,
                "name": COND_NAMES[cid],
                "color": COND_COLORS[cid],
                "arms": arm_data,
                "total_pulls": sum(a["pulls"] for a in arms),
                "exploration_ratio": compute_exploration_ratio(conn, cid),
                "preset_entropy": compute_preset_entropy(conn, cid),
                "cumulative_regret": regret["regret"],
            }
        )
    return bandits


def _differentiator_check(conn):
    c3_fb = conn.execute(
        "SELECT COUNT(*) FROM feedback f"
        " JOIN episodes e ON f.episode_id = e.episode_id"
        " WHERE e.condition_id = 3"
    ).fetchone()[0]
    c3_emb = conn.execute(
        "SELECT COUNT(*) FROM feedback_embeddings fe"
        " JOIN feedback f ON fe.feedback_id = f.feedback_id"
        " JOIN episodes e ON f.episode_id = e.episode_id"
        " WHERE e.condition_id = 3"
    ).fetchone()[0]
    c2_emb = conn.execute(
        "SELECT COUNT(*) FROM feedback_embeddings fe"
        " JOIN feedback f ON fe.feedback_id = f.feedback_id"
        " JOIN episodes e ON f.episode_id = e.episode_id"
        " WHERE e.condition_id = 2"
    ).fetchone()[0]
    return {"c3_embeddings": c3_emb, "c3_feedback": c3_fb, "c2_embeddings": c2_emb}


def _recent_episodes(conn, limit=10, task_ids=None):
    if task_ids:
        ph = ",".join("?" * len(task_ids))
        rows = conn.execute(
            f"""SELECT e.condition_id, e.task_id, e.preset_id, e.step_count,
                       e.total_tokens, e.completed_at, t.title, t.theme,
                       f.rating_recency, f.rating_importance, f.rating_relevance,
                       f.inferred_recency, f.inferred_importance, f.inferred_relevance
                FROM episodes e
                LEFT JOIN feedback f ON e.episode_id = f.episode_id
                LEFT JOIN tasks t ON e.task_id = t.task_id
                WHERE e.task_id IN ({ph})
                ORDER BY e.episode_id DESC LIMIT ?""",
            task_ids + [limit],
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT e.condition_id, e.task_id, e.preset_id, e.step_count,
                      e.total_tokens, e.completed_at, t.title, t.theme,
                      f.rating_recency, f.rating_importance, f.rating_relevance,
                      f.inferred_recency, f.inferred_importance, f.inferred_relevance
               FROM episodes e
               LEFT JOIN feedback f ON e.episode_id = f.episode_id
               LEFT JOIN tasks t ON e.task_id = t.task_id
               ORDER BY e.episode_id DESC LIMIT ?""",
            (limit,),
        ).fetchall()

    recent = []
    for r in rows:
        fb = None
        if r["rating_recency"] is not None:
            fb = {
                "type": "likert",
                "r": r["rating_recency"],
                "i": r["rating_importance"],
                "v": r["rating_relevance"],
            }
        elif r["inferred_recency"] is not None:
            fb = {
                "type": "inferred",
                "r": round(r["inferred_recency"], 2),
                "i": round(r["inferred_importance"], 2),
                "v": round(r["inferred_relevance"], 2),
            }
        recent.append(
            {
                "condition": COND_NAMES.get(r["condition_id"], "?"),
                "color": COND_COLORS.get(r["condition_id"], "#666"),
                "cid": r["condition_id"],
                "task_id": r["task_id"],
                "title": r["title"] or r["task_id"],
                "theme": r["theme"] if "theme" in r.keys() else "",
                "preset": r["preset_id"],
                "tokens": r["total_tokens"],
                "steps": r["step_count"],
                "time": r["completed_at"][-8:] if r["completed_at"] else "",
                "feedback": fb,
            }
        )
    return recent


# ── Route handlers ──


def get_overview_data(conn):
    data = {}
    data["total_episodes"] = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
    last_task = conn.execute(
        "SELECT value FROM experiment_metadata WHERE key='last_completed_task'"
    ).fetchone()
    data["tasks_done"] = int(last_task["value"]) if last_task else 0
    data["total_tasks"] = TOTAL_TASKS
    data["pct"] = round(data["tasks_done"] / TOTAL_TASKS * 100, 1)

    start_row = conn.execute(
        "SELECT value FROM experiment_metadata WHERE key='start_timestamp'"
    ).fetchone()
    if start_row and data["tasks_done"] > 0:
        start_time = datetime.strptime(start_row["value"], "%Y-%m-%dT%H:%M:%S")
        elapsed = (datetime.now() - start_time).total_seconds()
        data["elapsed_hours"] = round(elapsed / 3600, 1)
        rate = data["tasks_done"] / elapsed if elapsed > 0 else 0
        remaining = TOTAL_TASKS - data["tasks_done"]
        data["eta_hours"] = round(remaining / rate / 3600, 1) if rate > 0 else 0
        data["tasks_per_hour"] = round(rate * 3600, 1)
    else:
        data["elapsed_hours"] = 0
        data["eta_hours"] = 0
        data["tasks_per_hour"] = 0

    data["total_tokens"] = conn.execute(
        "SELECT COALESCE(SUM(total_tokens),0) FROM episodes"
    ).fetchone()[0]
    data["conditions"] = _condition_stats(conn)

    # Theme summaries
    themes = get_themes(conn)
    theme_summaries = []
    for theme in themes:
        tids = [
            r["task_id"]
            for r in conn.execute(
                "SELECT task_id FROM tasks WHERE theme=?",
                (theme,),
            ).fetchall()
        ]
        if not tids:
            continue
        ph = ",".join("?" * len(tids))
        ep_count = conn.execute(
            f"SELECT COUNT(*) FROM episodes WHERE task_id IN ({ph})",
            tids,
        ).fetchone()[0]
        total_possible = len(tids) * 4
        theme_summaries.append(
            {
                "theme": theme,
                "tasks": len(tids),
                "episodes": ep_count,
                "total_possible": total_possible,
                "pct": round(ep_count / total_possible * 100) if total_possible > 0 else 0,
            }
        )
    data["themes"] = theme_summaries
    data["bandits"] = _bandit_state(conn)
    data.update(_differentiator_check(conn))

    # Reward drift per condition
    data["drift"] = {}
    for cid in [2, 3, 4]:
        data["drift"][str(cid)] = compute_reward_drift(conn, cid)

    # Efficiency comparison
    ctrl_steps = (
        conn.execute("SELECT SUM(step_count) FROM episodes WHERE condition_id=1").fetchone()[0] or 0
    )
    data["efficiency"] = {}
    for cid in [2, 3, 4]:
        cond_steps = (
            conn.execute(
                "SELECT SUM(step_count) FROM episodes WHERE condition_id=?",
                (cid,),
            ).fetchone()[0]
            or 0
        )
        savings = round((ctrl_steps - cond_steps) / ctrl_steps * 100, 1) if ctrl_steps > 0 else 0
        data["efficiency"][str(cid)] = {"steps": cond_steps, "savings_pct": savings}

    # Cost estimate: $1.00 per 1M tokens (MiniMax M2.5 blended rate)
    total_tokens = data["total_tokens"]
    data["cost_total"] = round(total_tokens / 1_000_000 * 1.0, 2)
    data["cost_per_episode"] = (
        round(data["cost_total"] / data["total_episodes"], 4) if data["total_episodes"] > 0 else 0
    )

    # Parse rate trend
    data["parse_rate_trend"] = compute_parse_rate_trend(conn)

    # Completion rate with projection
    data["completion_rate"] = compute_completion_rate(conn)

    data["recent"] = _recent_episodes(conn, limit=12)
    return data


def get_theme_data(conn, theme):
    data = {"theme": theme}
    tids = [
        r["task_id"]
        for r in conn.execute(
            "SELECT task_id FROM tasks WHERE theme=?",
            (theme,),
        ).fetchall()
    ]
    if not tids:
        data["conditions"] = []
        data["recent"] = []
        return data

    data["total_tasks"] = len(tids)
    ph = ",".join("?" * len(tids))

    conditions = []
    for cid in range(1, 5):
        params = tids + [cid]
        row = conn.execute(
            f"SELECT COUNT(*) as n,"
            f" AVG(total_tokens) as avg_tok,"
            f" AVG(step_count) as avg_steps,"
            f" SUM(total_tokens) as sum_tok,"
            f" AVG(input_tokens) as avg_in_tok,"
            f" AVG(output_tokens) as avg_out_tok"
            f" FROM episodes"
            f" WHERE task_id IN ({ph}) AND condition_id=?",
            params,
        ).fetchone()
        fb_count = conn.execute(
            f"SELECT COUNT(*) FROM feedback f JOIN episodes e ON f.episode_id=e.episode_id "
            f"WHERE e.task_id IN ({ph}) AND e.condition_id=?",
            params,
        ).fetchone()[0]
        avg_r = conn.execute(
            f"""SELECT AVG(COALESCE(
                    (rating_recency+rating_importance+rating_relevance-3)/12.0,
                    (inferred_recency+inferred_importance+inferred_relevance)/3.0
                )) as v FROM feedback f JOIN episodes e ON f.episode_id=e.episode_id
                WHERE e.task_id IN ({ph}) AND e.condition_id=?""",
            params,
        ).fetchone()["v"]
        dim_row = conn.execute(
            f"""SELECT
                 AVG(COALESCE((rating_recency-1)/4.0,inferred_recency)) as r,
                 AVG(COALESCE((rating_importance-1)/4.0,inferred_importance)) as i,
                 AVG(COALESCE((rating_relevance-1)/4.0,inferred_relevance)) as v
                FROM feedback f JOIN episodes e ON f.episode_id=e.episode_id
                WHERE e.task_id IN ({ph}) AND e.condition_id=?""",
            params,
        ).fetchone()
        preset_rows = conn.execute(
            f"SELECT preset_id, COUNT(*) as cnt"
            f" FROM episodes"
            f" WHERE task_id IN ({ph}) AND condition_id=?"
            f" GROUP BY preset_id ORDER BY cnt DESC",
            params,
        ).fetchall()

        ir = compute_ndcg_mrr(conn, cid, tids)
        n = row["n"] or 0
        conditions.append(
            {
                "id": cid,
                "name": COND_NAMES[cid],
                "short": COND_SHORT[cid],
                "color": COND_COLORS[cid],
                "episodes": n,
                "parse_rate": round(fb_count / n * 100) if n > 0 else 0,
                "avg_reward": round(avg_r, 3) if avg_r else None,
                "avg_tokens": round(row["avg_tok"]) if row["avg_tok"] else 0,
                "avg_input_tokens": round(row["avg_in_tok"]) if row["avg_in_tok"] else 0,
                "avg_output_tokens": round(row["avg_out_tok"]) if row["avg_out_tok"] else 0,
                "avg_steps": round(row["avg_steps"], 1) if row["avg_steps"] else 0,
                "total_tokens": row["sum_tok"] or 0,
                "avg_recency": round(dim_row["r"], 3) if dim_row["r"] else None,
                "avg_importance": round(dim_row["i"], 3) if dim_row["i"] else None,
                "avg_relevance": round(dim_row["v"], 3) if dim_row["v"] else None,
                "ndcg5": ir["ndcg5"],
                "mrr": ir["mrr"],
                "presets": [{"id": r["preset_id"], "count": r["cnt"]} for r in preset_rows],
            }
        )
    data["conditions"] = conditions
    data["recent"] = _recent_episodes(conn, limit=8, task_ids=tids)
    return data


def get_condition_detail(conn, condition_id):
    """Deep dive into a single condition's trends."""
    data = {"condition_id": condition_id, "name": COND_NAMES.get(condition_id, "?")}
    data["reward_trend"] = compute_reward_trend(conn, condition_id)
    data["dim_trends"] = compute_dimension_trends(conn, condition_id)
    data["step_trend"] = compute_step_trend(conn, condition_id)

    if condition_id > 1:
        regret = compute_cumulative_regret(conn, condition_id)
        data["regret_points"] = regret["points"]
        data["drift"] = compute_reward_drift(conn, condition_id)
    return data


# ── Routes ──


@app.route("/api/runs")
def api_runs():
    return jsonify(list_runs())


@app.route("/api/overview")
def api_overview():
    conn = get_db(request.args.get("db", "experiment_v3.db"))
    if not conn:
        return jsonify({"error": "no db"})
    data = get_overview_data(conn)
    conn.close()
    return jsonify(data)


@app.route("/api/theme")
def api_theme():
    conn = get_db(request.args.get("db", "experiment_v3.db"))
    if not conn:
        return jsonify({"error": "no db"})
    data = get_theme_data(conn, request.args.get("theme", ""))
    conn.close()
    return jsonify(data)


@app.route("/api/trends")
def api_trends():
    conn = get_db(request.args.get("db", "experiment_v3.db"))
    if not conn:
        return jsonify({"error": "no db"})
    data = {}
    for cid in range(1, 5):
        data[str(cid)] = {
            "reward_trend": compute_reward_trend(conn, cid),
            "step_trend": compute_step_trend(conn, cid),
            "token_trend": compute_token_trend(conn, cid),
            "input_token_trend": compute_input_token_trend(conn, cid),
            "output_token_trend": compute_output_token_trend(conn, cid),
            "dim_trends": compute_dimension_trends(conn, cid),
        }
    conn.close()
    return jsonify(data)


@app.route("/api/theme-trends")
def api_theme_trends():
    conn = get_db(request.args.get("db", "experiment_v3.db"))
    if not conn:
        return jsonify({"error": "no db"})
    theme = request.args.get("theme", "")
    tids = [
        r["task_id"]
        for r in conn.execute("SELECT task_id FROM tasks WHERE theme=?", (theme,)).fetchall()
    ]
    if not tids:
        conn.close()
        return jsonify({"error": "no tasks for theme"})
    data = {}
    for cid in range(1, 5):
        data[str(cid)] = {
            "reward_trend": compute_reward_trend(conn, cid, task_ids=tids),
            "step_trend": compute_step_trend(conn, cid, task_ids=tids),
            "token_trend": compute_token_trend(conn, cid, task_ids=tids),
            "dim_trends": compute_dimension_trends(conn, cid, task_ids=tids),
        }
    conn.close()
    return jsonify(data)


@app.route("/api/condition")
def api_condition():
    conn = get_db(request.args.get("db", "experiment_v3.db"))
    if not conn:
        return jsonify({"error": "no db"})
    cid = int(request.args.get("cid", 1))
    data = get_condition_detail(conn, cid)
    conn.close()
    return jsonify(data)


@app.route("/api/bandit-history")
def api_bandit_history():
    conn = get_db(request.args.get("db", "experiment_v3.db"))
    if not conn:
        return jsonify({"error": "no db"})
    data = {}
    for cid in [2, 3, 4]:
        data[str(cid)] = compute_bandit_history(conn, cid)
    conn.close()
    return jsonify(data)


@app.route("/api/stream")
def stream():
    db_file = request.args.get("db", "experiment_v3.db")

    def generate():
        while True:
            conn = get_db(db_file)
            if conn:
                data = get_overview_data(conn)
                conn.close()
            else:
                data = {"error": "no db"}
            yield f"data: {json.dumps(data)}\n\n"
            time.sleep(15)

    return Response(generate(), mimetype="text/event-stream")


@app.route("/")
def index():
    return HTML


# ── HTML (single-page app) ──
# The HTML is loaded from an external file to keep this file manageable
HTML_PATH = ROOT / "scripts" / "dashboard.html"


def _load_html():
    if HTML_PATH.exists():
        return HTML_PATH.read_text()
    return "<h1>dashboard.html not found</h1>"


# Lazy load
HTML = None


@app.before_request
def ensure_html():
    global HTML
    if HTML is None:
        HTML = _load_html()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    HTML = _load_html()
    print(f"Dashboard: http://localhost:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
