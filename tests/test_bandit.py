"""Tests for Thompson Sampling bandit."""

import numpy as np
import pytest

from src.db.schema import init_bandit_state, init_db, seed_weight_presets
from src.experiment.bandit import (
    ThompsonSamplingBandit,
    cost_aware_reward,
    normalize_likert_to_reward,
)

PRESETS = {
    "equal": [0.333, 0.333, 0.334],
    "relevance_heavy": [0.10, 0.20, 0.70],
    "recency_heavy": [0.70, 0.10, 0.20],
}


@pytest.fixture
def db_with_bandit():
    """Database with presets and initialized bandit state."""
    conn = init_db(":memory:")
    seed_weight_presets(conn, PRESETS)
    init_bandit_state(conn, [2, 3], list(PRESETS.keys()))
    yield conn
    conn.close()


class TestArmSelection:
    def test_select_returns_valid_preset(self, db_with_bandit):
        bandit = ThompsonSamplingBandit(
            db_with_bandit, condition_id=2, rng=np.random.default_rng(42)
        )
        selected = bandit.select_arm()
        assert selected in PRESETS

    def test_all_arms_get_selected_eventually(self, db_with_bandit):
        """With uniform priors, all arms should be explored."""
        bandit = ThompsonSamplingBandit(
            db_with_bandit, condition_id=2, rng=np.random.default_rng(42)
        )
        selected = set()
        for _ in range(100):
            selected.add(bandit.select_arm())
        assert selected == set(PRESETS.keys())

    def test_conditions_are_independent(self, db_with_bandit):
        """Updating condition 2 shouldn't affect condition 3."""
        bandit2 = ThompsonSamplingBandit(
            db_with_bandit, condition_id=2, rng=np.random.default_rng(42)
        )
        bandit3 = ThompsonSamplingBandit(
            db_with_bandit, condition_id=3, rng=np.random.default_rng(42)
        )

        # Update condition 2 heavily
        for _ in range(50):
            bandit2.update("relevance_heavy", 0.95)

        # Condition 3 should still have uniform priors
        arms3 = bandit3.get_arms()
        for arm in arms3:
            assert arm.alpha == 1.0
            assert arm.beta == 1.0


class TestArmUpdate:
    def test_update_increments_alpha(self, db_with_bandit):
        bandit = ThompsonSamplingBandit(db_with_bandit, condition_id=2)
        bandit.update("equal", 0.8)

        arms = bandit.get_arms()
        equal_arm = next(a for a in arms if a.preset_id == "equal")
        assert equal_arm.alpha == 1.0 + 0.8
        assert equal_arm.beta == 1.0 + 0.2
        assert equal_arm.pulls == 1

    def test_update_rejects_invalid_reward(self, db_with_bandit):
        bandit = ThompsonSamplingBandit(db_with_bandit, condition_id=2)
        with pytest.raises(ValueError, match="Reward must be in"):
            bandit.update("equal", 1.5)

    def test_convergence_toward_best_arm(self, db_with_bandit):
        """After many high-reward updates, one arm should dominate."""
        bandit = ThompsonSamplingBandit(
            db_with_bandit, condition_id=2, rng=np.random.default_rng(42)
        )

        # Simulate: relevance_heavy consistently gets high reward
        for _ in range(100):
            bandit.update("relevance_heavy", 0.9)
            bandit.update("equal", 0.4)
            bandit.update("recency_heavy", 0.3)

        arms = bandit.get_arms()
        best = max(arms, key=lambda a: a.mean)
        assert best.preset_id == "relevance_heavy"
        assert best.mean > 0.8


class TestSummary:
    def test_summary_structure(self, db_with_bandit):
        bandit = ThompsonSamplingBandit(db_with_bandit, condition_id=2)
        summary = bandit.get_summary()

        assert summary["condition_id"] == 2
        assert len(summary["arms"]) == 3
        assert summary["total_pulls"] == 0
        assert summary["best_arm"] is not None


class TestRewardNormalization:
    def test_min_likert(self):
        reward = normalize_likert_to_reward(1, 1, 1)
        assert reward == 0.0

    def test_max_likert(self):
        reward = normalize_likert_to_reward(5, 5, 5)
        assert reward == 1.0

    def test_mid_likert(self):
        reward = normalize_likert_to_reward(3, 3, 3)
        assert reward == pytest.approx(0.5)

    def test_mixed_ratings(self):
        reward = normalize_likert_to_reward(5, 1, 3)
        # avg = 3.0, normalized = (3 - 1) / 4 = 0.5
        assert reward == pytest.approx(0.5)


class TestCostAwareReward:
    def test_cost_aware_same_tokens_same_reward(self):
        """When tokens == baseline, reward is unchanged."""
        assert cost_aware_reward(0.6, 2500, 2500) == pytest.approx(0.6)

    def test_cost_aware_fewer_tokens_higher_reward(self):
        """Fewer tokens → higher effective reward."""
        raw = 0.5
        result = cost_aware_reward(raw, 1250, 2500)
        # 0.5 * (2500 / 1250) = 1.0
        assert result == pytest.approx(1.0)
        assert result >= raw

    def test_cost_aware_more_tokens_lower_reward(self):
        """More tokens → lower effective reward."""
        raw = 0.8
        result = cost_aware_reward(raw, 5000, 2500)
        # 0.8 * (2500 / 5000) = 0.4
        assert result == pytest.approx(0.4)
        assert result < raw

    def test_cost_aware_clamp(self):
        """Result always in [0, 1]."""
        # Very efficient: 0.9 * (2500 / 500) = 4.5 → clamped to 1.0
        assert cost_aware_reward(0.9, 500, 2500) == 1.0
        # Zero reward stays zero regardless of efficiency
        assert cost_aware_reward(0.0, 500, 2500) == 0.0

    def test_cost_aware_zero_tokens_safe(self):
        """Division by zero handled — pass through raw reward."""
        assert cost_aware_reward(0.7, 0, 2500) == 0.7


class TestDiscountedThompsonSampling:
    """Tests for Discounted Thompson Sampling (drift robustness)."""

    def test_discount_none_no_change(self, db_with_bandit):
        """Without discount, behavior is identical to standard TS."""
        bandit = ThompsonSamplingBandit(db_with_bandit, condition_id=2)
        bandit.update("equal", 0.8)

        arms = bandit.get_arms()
        equal_arm = next(a for a in arms if a.preset_id == "equal")
        # Standard update: alpha = 1.0 + 0.8, beta = 1.0 + 0.2
        assert equal_arm.alpha == pytest.approx(1.8)
        assert equal_arm.beta == pytest.approx(1.2)

    def test_discount_reduces_alpha_beta(self, db_with_bandit):
        """With discount, alpha and beta are decayed before the new observation."""
        discount = 0.9
        bandit = ThompsonSamplingBandit(
            db_with_bandit,
            condition_id=2,
            discount=discount,
        )
        bandit.update("equal", 0.8)

        arms = bandit.get_arms()
        equal_arm = next(a for a in arms if a.preset_id == "equal")
        # Discounted: alpha = 1.0 * 0.9 + 0.8 = 1.7
        #             beta  = 1.0 * 0.9 + 0.2 = 1.1
        assert equal_arm.alpha == pytest.approx(1.7)
        assert equal_arm.beta == pytest.approx(1.1)

    def test_discount_enables_drift_adaptation(self, db_with_bandit):
        """Discounted TS adapts faster than undiscounted when the best arm shifts."""
        rng_seed = 99

        # --- Undiscounted bandit ---
        bandit_std = ThompsonSamplingBandit(
            db_with_bandit,
            condition_id=2,
            rng=np.random.default_rng(rng_seed),
        )
        # Phase 1: "equal" is best (50 rounds)
        for _ in range(50):
            bandit_std.update("equal", 0.9)
            bandit_std.update("relevance_heavy", 0.3)
        # Phase 2: "relevance_heavy" becomes best (50 rounds)
        for _ in range(50):
            bandit_std.update("equal", 0.3)
            bandit_std.update("relevance_heavy", 0.9)

        arms_std = bandit_std.get_arms()
        equal_std = next(a for a in arms_std if a.preset_id == "equal")
        relev_std = next(a for a in arms_std if a.preset_id == "relevance_heavy")

        # --- Discounted bandit (separate condition to avoid interference) ---
        bandit_disc = ThompsonSamplingBandit(
            db_with_bandit,
            condition_id=3,
            rng=np.random.default_rng(rng_seed),
            discount=0.95,
        )
        for _ in range(50):
            bandit_disc.update("equal", 0.9)
            bandit_disc.update("relevance_heavy", 0.3)
        for _ in range(50):
            bandit_disc.update("equal", 0.3)
            bandit_disc.update("relevance_heavy", 0.9)

        arms_disc = bandit_disc.get_arms()
        equal_disc = next(a for a in arms_disc if a.preset_id == "equal")
        relev_disc = next(a for a in arms_disc if a.preset_id == "relevance_heavy")

        # After the shift, discounted bandit should favor relevance_heavy more strongly
        disc_gap = relev_disc.mean - equal_disc.mean
        std_gap = relev_std.mean - equal_std.mean
        assert disc_gap > std_gap, (
            f"Discounted bandit should adapt faster. disc_gap={disc_gap:.4f}, std_gap={std_gap:.4f}"
        )

    def test_discount_preserves_ratio(self, db_with_bandit):
        """Discounting alpha and beta by the same factor preserves alpha/(alpha+beta)."""
        # Set up a known state: update several times without discount first
        bandit = ThompsonSamplingBandit(db_with_bandit, condition_id=2)
        bandit.update("equal", 0.8)
        bandit.update("equal", 0.6)
        bandit.update("equal", 0.7)

        arms_before = bandit.get_arms()
        equal_before = next(a for a in arms_before if a.preset_id == "equal")
        ratio_before = equal_before.alpha / (equal_before.alpha + equal_before.beta)

        # Now apply discount directly via SQL (simulating what update() does internally)
        discount = 0.95
        db_with_bandit.execute(
            "UPDATE bandit_state SET alpha = alpha * ?, beta = beta * ? "
            "WHERE condition_id = ? AND preset_id = ?",
            (discount, discount, 2, "equal"),
        )
        db_with_bandit.commit()

        arms_after = bandit.get_arms()
        equal_after = next(a for a in arms_after if a.preset_id == "equal")
        ratio_after = equal_after.alpha / (equal_after.alpha + equal_after.beta)

        assert ratio_after == pytest.approx(ratio_before, abs=1e-10)
        # But the absolute values should be smaller
        assert equal_after.alpha < equal_before.alpha
        assert equal_after.beta < equal_before.beta

    def test_discount_invalid_value_rejected(self, db_with_bandit):
        """Discount outside (0, 1) raises ValueError."""
        with pytest.raises(ValueError, match="Discount must be in"):
            ThompsonSamplingBandit(db_with_bandit, condition_id=2, discount=1.0)
        with pytest.raises(ValueError, match="Discount must be in"):
            ThompsonSamplingBandit(db_with_bandit, condition_id=2, discount=0.0)
        with pytest.raises(ValueError, match="Discount must be in"):
            ThompsonSamplingBandit(db_with_bandit, condition_id=2, discount=1.5)
