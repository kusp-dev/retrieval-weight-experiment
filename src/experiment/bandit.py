"""
Thompson Sampling bandit for retrieval weight selection.

Each condition maintains independent Beta(α, β) posteriors over N weight presets (12 in v3).
The reward signal comes from dimension-specific LLM self-assessment (Likert 1-5),
normalized to [0, 1].

Key design choices (from learnings):
  - discrete presets (12 in v3), not continuous optimization
  - Rate retrieval dimensions (external inputs), not output quality
  - Beta distribution: α += reward, β += (1 - reward)
"""

import sqlite3
from dataclasses import dataclass

import numpy as np


@dataclass
class BanditArm:
    """State of one arm (weight preset) in the bandit."""

    preset_id: str
    alpha: float
    beta: float
    pulls: int
    total_reward: float

    @property
    def mean(self) -> float:
        """Expected value of the Beta distribution."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        """Variance of the Beta distribution."""
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))


class ThompsonSamplingBandit:
    """Thompson Sampling over retrieval weight presets.

    Each condition_id gets independent posteriors. Control condition (1)
    doesn't use this — it always returns 'equal'.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        condition_id: int,
        rng: np.random.Generator | None = None,
        discount: float | None = None,
    ):
        if discount is not None and not (0.0 < discount < 1.0):
            raise ValueError(f"Discount must be in (0, 1), got {discount}")
        self.conn = conn
        self.condition_id = condition_id
        self.rng = rng or np.random.default_rng()
        self.discount = discount

    def get_arms(self) -> list[BanditArm]:
        """Load current state of all arms for this condition."""
        rows = self.conn.execute(
            """SELECT preset_id, alpha, beta, pulls, total_reward
               FROM bandit_state
               WHERE condition_id = ?
               ORDER BY preset_id""",
            (self.condition_id,),
        ).fetchall()

        return [
            BanditArm(
                preset_id=row["preset_id"],
                alpha=row["alpha"],
                beta=row["beta"],
                pulls=row["pulls"],
                total_reward=row["total_reward"],
            )
            for row in rows
        ]

    def select_arm(self) -> str:
        """Sample from each arm's Beta posterior and select the highest.

        Returns the preset_id of the selected arm.
        """
        arms = self.get_arms()
        if not arms:
            raise ValueError(
                f"No bandit arms found for condition {self.condition_id}. "
                "Did you call init_bandit_state()?"
            )

        samples = [(arm.preset_id, self.rng.beta(arm.alpha, arm.beta)) for arm in arms]

        selected = max(samples, key=lambda x: x[1])
        return selected[0]

    def update(self, preset_id: str, reward: float) -> None:
        """Update the Beta posterior for the selected arm.

        Args:
            preset_id: which arm was pulled
            reward: normalized reward in [0, 1]
        """
        if not 0.0 <= reward <= 1.0:
            raise ValueError(f"Reward must be in [0, 1], got {reward}")

        # Discounted Thompson Sampling: decay old observations before adding new one.
        # This lets the bandit adapt when the optimal arm shifts over time.
        if self.discount is not None:
            self.conn.execute(
                """UPDATE bandit_state
                   SET alpha = alpha * ?, beta = beta * ?
                   WHERE condition_id = ? AND preset_id = ?""",
                (self.discount, self.discount, self.condition_id, preset_id),
            )

        self.conn.execute(
            """UPDATE bandit_state
               SET alpha = alpha + ?,
                   beta = beta + ?,
                   pulls = pulls + 1,
                   total_reward = total_reward + ?,
                   last_updated = datetime('now')
               WHERE condition_id = ? AND preset_id = ?""",
            (reward, 1.0 - reward, reward, self.condition_id, preset_id),
        )
        self.conn.commit()

    def get_summary(self) -> dict:
        """Return a summary of bandit state for logging/analysis."""
        arms = self.get_arms()
        return {
            "condition_id": self.condition_id,
            "arms": [
                {
                    "preset_id": arm.preset_id,
                    "alpha": arm.alpha,
                    "beta": arm.beta,
                    "mean": round(arm.mean, 4),
                    "pulls": arm.pulls,
                }
                for arm in arms
            ],
            "best_arm": max(arms, key=lambda a: a.mean).preset_id if arms else None,
            "total_pulls": sum(a.pulls for a in arms),
        }


def cost_aware_reward(raw_reward: float, tokens_used: int, baseline_tokens: int = 2500) -> float:
    """Scale reward by token efficiency.

    reward_effective = raw_reward * (baseline_tokens / tokens_used)
    Clamped to [0, 1] to stay within Beta distribution bounds.

    baseline_tokens: expected tokens for a typical episode. Episodes using
    fewer tokens get a reward boost; more tokens get penalized.
    """
    if tokens_used <= 0:
        return raw_reward  # degenerate case — pass through unchanged
    effective = raw_reward * (baseline_tokens / tokens_used)
    return max(0.0, min(1.0, effective))


def normalize_likert_to_reward(
    rating_recency: int,
    rating_importance: int,
    rating_relevance: int,
) -> float:
    """Convert Likert ratings (1-5) to a single [0, 1] reward signal.

    Uses the average of all three dimension ratings, normalized from
    [1, 5] to [0, 1]. This is the reward that updates the bandit posterior
    for whichever weight preset was used.

    The individual dimension ratings are stored separately for analysis —
    this function only produces the bandit's reward signal.
    """
    avg = (rating_recency + rating_importance + rating_relevance) / 3.0
    return (avg - 1.0) / 4.0  # Map [1, 5] → [0, 1]
