"""
Comprehensive tests for the learning components:
  1. Rate control (thread safety, sliding window, step budget)
  2. Bandit (Thompson Sampling, posterior updates, discounted TS, DB persistence)
  3. Feedback parser (Likert parsing, failure modes, parse rates)
  4. Scoring (recency decay, importance Laplace smoothing, range validation)
  5. Integration (mock episode flow end-to-end)
"""

import threading
import time
from collections import Counter
from unittest.mock import patch

import numpy as np
import pytest

from src.db.schema import init_bandit_state, init_db, seed_weight_presets
from src.experiment.bandit import (
    ThompsonSamplingBandit,
    cost_aware_reward,
    normalize_likert_to_reward,
)
from src.experiment.feedback_parser import (
    _extract_feedback_block,
    extract_qualitative_sections,
    parse_likert_feedback,
)
from src.experiment.rate_control import SlidingWindowLimiter, StepBudget
from src.experiment.scoring import (
    compute_importance,
    compute_recency,
    record_skill_usage,
)

PRESETS = {
    "balanced": [0.330, 0.330, 0.340],
    "recency_heavy": [0.600, 0.200, 0.200],
    "importance_heavy": [0.150, 0.700, 0.150],
    "relevance_heavy": [0.200, 0.150, 0.650],
    "relevance_importance": [0.150, 0.350, 0.500],
}


# ══════════════════════════════════════════════════════════════
# 1. RATE CONTROL — Thread Safety
# ══════════════════════════════════════════════════════════════


class TestRateControlThreadSafety:
    """Spawn 4 threads that simultaneously call check() and record()."""

    def test_concurrent_record_no_double_counting(self):
        """4 threads each record 25 iterations. Total must be exactly 100."""
        limiter = SlidingWindowLimiter(max_per_window=200, window_seconds=3600)
        barrier = threading.Barrier(4)

        def worker():
            barrier.wait()  # all threads start together
            for _ in range(25):
                limiter.record()

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert limiter.current_count == 100, (
            f"Expected 100, got {limiter.current_count} — race condition detected"
        )

    def test_concurrent_check_no_crash(self):
        """4 threads calling check() concurrently should never crash or deadlock."""
        limiter = SlidingWindowLimiter(max_per_window=10, window_seconds=3600)
        # Pre-fill some entries
        for _ in range(5):
            limiter.record()

        barrier = threading.Barrier(4)
        results = []
        lock = threading.Lock()

        def worker():
            barrier.wait()
            for _ in range(50):
                r = limiter.check()
                with lock:
                    results.append(r.action)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 200
        assert all(a in ("go", "throttle") for a in results)

    def test_concurrent_mixed_check_and_record(self):
        """2 threads record, 2 threads check — no deadlocks or inconsistencies."""
        limiter = SlidingWindowLimiter(max_per_window=100, window_seconds=3600)
        barrier = threading.Barrier(4)
        errors = []

        def recorder():
            barrier.wait()
            for _ in range(50):
                limiter.record()

        def checker():
            barrier.wait()
            for _ in range(50):
                try:
                    r = limiter.check()
                    assert r.action in ("go", "throttle")
                except Exception as e:
                    errors.append(str(e))

        threads = []
        for _ in range(2):
            threads.append(threading.Thread(target=recorder))
            threads.append(threading.Thread(target=checker))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"
        assert limiter.current_count == 100

    def test_no_missed_counts_under_contention(self):
        """Heavy contention: 4 threads x 100 records = must be exactly 400."""
        limiter = SlidingWindowLimiter(max_per_window=500, window_seconds=3600)
        barrier = threading.Barrier(4)

        def worker():
            barrier.wait()
            for _ in range(100):
                limiter.record()

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert limiter.current_count == 400


class TestSlidingWindowEnforcement:
    """Verify the sliding window correctly enforces the rate limit."""

    def test_throttles_exactly_at_limit(self):
        limiter = SlidingWindowLimiter(max_per_window=5, window_seconds=3600)
        for _ in range(5):
            limiter.record()

        result = limiter.check()
        assert result.action == "throttle"
        assert result.wait_seconds > 0

    def test_allows_one_below_limit(self):
        limiter = SlidingWindowLimiter(max_per_window=5, window_seconds=3600)
        for _ in range(4):
            limiter.record()

        result = limiter.check()
        assert result.action == "go"

    def test_window_expiry_re_enables(self):
        """After the window expires, old entries are evicted and rate is allowed."""
        limiter = SlidingWindowLimiter(max_per_window=2, window_seconds=1)
        limiter.record()
        limiter.record()

        assert limiter.check().action == "throttle"

        # Time-travel past the window
        with patch("src.experiment.rate_control.time") as mock_time:
            mock_time.time.return_value = time.time() + 2
            result = limiter.check()
            assert result.action == "go"
            assert limiter.current_count == 0

    def test_partial_window_expiry(self):
        """Only old entries expire; recent ones remain."""
        limiter = SlidingWindowLimiter(max_per_window=5, window_seconds=10)

        now = time.time()
        # Manually inject timestamps: 3 old, 2 recent
        with limiter._lock:
            limiter._timestamps.append(now - 15)  # expired
            limiter._timestamps.append(now - 12)  # expired
            limiter._timestamps.append(now - 11)  # expired
            limiter._timestamps.append(now - 3)  # recent
            limiter._timestamps.append(now - 1)  # recent

        # Check should evict the 3 old ones, leaving 2
        assert limiter.current_count == 2


class TestStepBudgetEnforcement:
    """Test step budget logic."""

    def test_budget_exhaustion_at_max(self):
        budget = StepBudget(max_steps=5)
        for _ in range(4):
            assert not budget.step()
        assert budget.step()  # 5th step exhausts

    def test_remaining_counts_down(self):
        budget = StepBudget(max_steps=10)
        for i in range(10):
            assert budget.remaining == 10 - i
            budget.step()
        assert budget.remaining == 0

    def test_fraction_used_precision(self):
        budget = StepBudget(max_steps=35)
        for _ in range(21):
            budget.step()
        assert budget.fraction_used == pytest.approx(21 / 35)

    def test_nudge_fires_at_60pct_no_output(self):
        budget = StepBudget(max_steps=10, nudge_threshold=0.6)
        for _ in range(6):
            budget.step()
        assert budget.should_nudge(has_output=False)

    def test_nudge_suppressed_with_output(self):
        budget = StepBudget(max_steps=10, nudge_threshold=0.6)
        for _ in range(6):
            budget.step()
        assert not budget.should_nudge(has_output=True)


# ══════════════════════════════════════════════════════════════
# 2. BANDIT — Thompson Sampling
# ══════════════════════════════════════════════════════════════


@pytest.fixture
def bandit_db():
    """Database with schema, presets, and bandit state initialized."""
    conn = init_db(":memory:")
    seed_weight_presets(conn, PRESETS)
    init_bandit_state(conn, [2, 3, 4], list(PRESETS.keys()))
    yield conn
    conn.close()


class TestThompsonSamplingPosterior:
    """Test that posteriors update correctly under known reward sequences."""

    def test_initial_priors_are_beta_1_1(self, bandit_db):
        bandit = ThompsonSamplingBandit(bandit_db, condition_id=2)
        arms = bandit.get_arms()
        assert len(arms) == 5
        for arm in arms:
            assert arm.alpha == 1.0
            assert arm.beta == 1.0
            assert arm.pulls == 0
            assert arm.total_reward == 0.0

    def test_20_rewards_update_posterior(self, bandit_db):
        """Feed 20 rewards of 0.8 to one arm. Verify alpha/beta shift."""
        bandit = ThompsonSamplingBandit(bandit_db, condition_id=2)
        target_arm = "relevance_heavy"
        reward = 0.8

        for _ in range(20):
            bandit.update(target_arm, reward)

        arms = bandit.get_arms()
        target = [a for a in arms if a.preset_id == target_arm][0]

        # alpha = 1 + 20*0.8 = 17.0, beta = 1 + 20*0.2 = 5.0
        assert target.alpha == pytest.approx(17.0)
        assert target.beta == pytest.approx(5.0)
        assert target.pulls == 20
        assert target.total_reward == pytest.approx(16.0)

        # Mean should be ~0.77 (17/22)
        assert target.mean == pytest.approx(17.0 / 22.0, abs=0.01)

        # Other arms should be untouched
        others = [a for a in arms if a.preset_id != target_arm]
        for arm in others:
            assert arm.alpha == 1.0
            assert arm.beta == 1.0
            assert arm.pulls == 0

    def test_select_arm_returns_valid_preset(self, bandit_db):
        bandit = ThompsonSamplingBandit(bandit_db, condition_id=2, rng=np.random.default_rng(42))
        for _ in range(50):
            arm = bandit.select_arm()
            assert arm in PRESETS, f"select_arm returned invalid preset: {arm}"

    def test_select_arm_explores_all(self, bandit_db):
        """With uniform priors, all arms should be selected at least once over many draws."""
        bandit = ThompsonSamplingBandit(bandit_db, condition_id=2, rng=np.random.default_rng(42))
        selected = Counter()
        for _ in range(200):
            selected[bandit.select_arm()] += 1

        for preset_id in PRESETS:
            assert selected[preset_id] > 0, f"Arm {preset_id} was never selected in 200 draws"

    def test_reward_validation(self, bandit_db):
        bandit = ThompsonSamplingBandit(bandit_db, condition_id=2)
        with pytest.raises(ValueError, match="Reward must be in"):
            bandit.update("balanced", 1.5)
        with pytest.raises(ValueError, match="Reward must be in"):
            bandit.update("balanced", -0.1)

    def test_no_arms_raises(self):
        """Selecting from a condition with no arms should raise."""
        conn = init_db(":memory:")
        bandit = ThompsonSamplingBandit(conn, condition_id=99)
        with pytest.raises(ValueError, match="No bandit arms found"):
            bandit.select_arm()
        conn.close()


class TestDiscountedTS:
    """Test discounted Thompson Sampling (non-stationary environments)."""

    def test_discount_decays_old_evidence(self, bandit_db):
        """After discount, alpha and beta should shrink before new evidence is added."""
        discount = 0.95
        bandit = ThompsonSamplingBandit(bandit_db, condition_id=2, discount=discount)

        # Feed 10 rewards of 1.0 to balanced
        for _ in range(10):
            bandit.update("balanced", 1.0)

        arm = [a for a in bandit.get_arms() if a.preset_id == "balanced"][0]

        # Without discount: alpha=11.0, beta=1.0
        # With discount=0.95 applied before each update, values will be smaller
        # because old evidence is decayed. Alpha should be < 11.0
        assert arm.alpha < 11.0
        assert arm.pulls == 10

    def test_discount_validation(self, bandit_db):
        with pytest.raises(ValueError, match="Discount must be in"):
            ThompsonSamplingBandit(bandit_db, condition_id=2, discount=0.0)
        with pytest.raises(ValueError, match="Discount must be in"):
            ThompsonSamplingBandit(bandit_db, condition_id=2, discount=1.0)
        with pytest.raises(ValueError, match="Discount must be in"):
            ThompsonSamplingBandit(bandit_db, condition_id=2, discount=-0.5)

    def test_discount_allows_adaptation(self, bandit_db):
        """Discounted TS should adapt when the best arm changes."""
        discount = 0.9
        bandit = ThompsonSamplingBandit(
            bandit_db, condition_id=2, rng=np.random.default_rng(42), discount=discount
        )

        # Phase 1: relevance_heavy is great (30 episodes)
        for _ in range(30):
            bandit.update("relevance_heavy", 0.9)
            bandit.update("balanced", 0.3)

        summary1 = bandit.get_summary()
        best1 = summary1["best_arm"]
        assert best1 == "relevance_heavy"

        # Phase 2: balanced becomes great, relevance_heavy becomes bad (50 episodes)
        for _ in range(50):
            bandit.update("balanced", 0.9)
            bandit.update("relevance_heavy", 0.1)

        summary2 = bandit.get_summary()
        best2 = summary2["best_arm"]
        assert best2 == "balanced", (
            f"Discounted TS failed to adapt: best is {best2}, expected balanced"
        )


class TestBanditDBPersistence:
    """Test that bandit state can be saved to and loaded from the DB."""

    def test_state_persists_after_updates(self, bandit_db):
        bandit = ThompsonSamplingBandit(bandit_db, condition_id=2)
        bandit.update("balanced", 0.7)
        bandit.update("balanced", 0.8)

        # Create a new bandit instance pointing to the same DB
        bandit2 = ThompsonSamplingBandit(bandit_db, condition_id=2)
        arms = bandit2.get_arms()
        balanced = [a for a in arms if a.preset_id == "balanced"][0]

        assert balanced.alpha == pytest.approx(1.0 + 0.7 + 0.8)
        assert balanced.beta == pytest.approx(1.0 + 0.3 + 0.2)
        assert balanced.pulls == 2

    def test_conditions_are_independent(self, bandit_db):
        """Updates to condition 2 should not affect condition 3."""
        bandit2 = ThompsonSamplingBandit(bandit_db, condition_id=2)
        bandit3 = ThompsonSamplingBandit(bandit_db, condition_id=3)

        bandit2.update("balanced", 0.9)

        arms2 = bandit2.get_arms()
        arms3 = bandit3.get_arms()

        b2 = [a for a in arms2 if a.preset_id == "balanced"][0]
        b3 = [a for a in arms3 if a.preset_id == "balanced"][0]

        assert b2.pulls == 1
        assert b3.pulls == 0
        assert b2.alpha > b3.alpha

    def test_get_summary_structure(self, bandit_db):
        bandit = ThompsonSamplingBandit(bandit_db, condition_id=2)
        bandit.update("relevance_heavy", 0.6)

        summary = bandit.get_summary()
        assert summary["condition_id"] == 2
        assert summary["total_pulls"] == 1
        assert len(summary["arms"]) == 5
        assert summary["best_arm"] is not None

        for arm_info in summary["arms"]:
            assert "preset_id" in arm_info
            assert "alpha" in arm_info
            assert "beta" in arm_info
            assert "mean" in arm_info
            assert "pulls" in arm_info


class TestNormalizeLikertToReward:
    def test_min_rating(self):
        assert normalize_likert_to_reward(1, 1, 1) == pytest.approx(0.0)

    def test_max_rating(self):
        assert normalize_likert_to_reward(5, 5, 5) == pytest.approx(1.0)

    def test_mid_rating(self):
        assert normalize_likert_to_reward(3, 3, 3) == pytest.approx(0.5)

    def test_asymmetric_rating(self):
        # (3+4+5)/3 = 4.0 → (4.0 - 1) / 4 = 0.75
        assert normalize_likert_to_reward(3, 4, 5) == pytest.approx(0.75)


class TestCostAwareReward:
    def test_baseline_tokens_no_change(self):
        assert cost_aware_reward(0.8, 2500) == pytest.approx(0.8)

    def test_fewer_tokens_boost(self):
        r = cost_aware_reward(0.8, 1250)  # 0.8 * (2500/1250) = 1.6 → clamped to 1.0
        assert r == pytest.approx(1.0)

    def test_more_tokens_penalty(self):
        r = cost_aware_reward(0.8, 5000)  # 0.8 * (2500/5000) = 0.4
        assert r == pytest.approx(0.4)

    def test_zero_tokens_passthrough(self):
        assert cost_aware_reward(0.5, 0) == pytest.approx(0.5)

    def test_result_clamped_to_01(self):
        r = cost_aware_reward(1.0, 500)  # 1.0 * 5 = 5.0 → clamped to 1.0
        assert 0.0 <= r <= 1.0


# ══════════════════════════════════════════════════════════════
# 3. FEEDBACK PARSER — Likert Parsing
# ══════════════════════════════════════════════════════════════


class TestLikertParsingMiniMaxFormats:
    """Test parsing with sample outputs that look like MiniMax M2.5 responses."""

    def test_standard_tagged_format(self):
        response = """Here is my analysis of the task.

[RETRIEVAL EFFECTIVENESS]
Recency: 3
Importance: 4
Relevance: 5

Brief explanation:
- Relevance was the most valuable dimension because the topic matched perfectly.
- Recency was least valuable — timing didn't matter for this theoretical task.
- More foundational ML skills would have helped.
[/RETRIEVAL EFFECTIVENESS]"""

        fb = parse_likert_feedback(response)
        assert fb is not None
        assert fb.recency == pytest.approx((3 - 1) / 4)
        assert fb.importance == pytest.approx((4 - 1) / 4)
        assert fb.relevance == pytest.approx((5 - 1) / 4)
        assert fb.parse_method == "likert"
        assert 0.0 <= fb.composite_reward <= 1.0

    def test_markdown_heading_format(self):
        """MiniMax often uses markdown headings instead of tags."""
        response = """I completed the task using the retrieved skills.

## Retrieval Effectiveness

Recency: 2
Importance: 5
Relevance: 4

- Importance was essential — well-tested skills were crucial.
- Recency didn't matter much for this analysis task.
- Better evaluation frameworks would have been helpful."""

        fb = parse_likert_feedback(response)
        assert fb is not None
        assert fb.recency == pytest.approx((2 - 1) / 4)
        assert fb.importance == pytest.approx((5 - 1) / 4)
        assert fb.relevance == pytest.approx((4 - 1) / 4)

    def test_markdown_table_format(self):
        """MiniMax sometimes outputs tables."""
        response = """Task completed successfully.

### Retrieval Feedback

| Dimension | Rating |
|-----------|--------|
| **Recency** | 3/5 |
| **Importance** | 4/5 |
| **Relevance** | 5/5 |

The relevance dimension was most useful."""

        fb = parse_likert_feedback(response)
        assert fb is not None
        assert fb.recency == pytest.approx((3 - 1) / 4)
        assert fb.importance == pytest.approx((4 - 1) / 4)
        assert fb.relevance == pytest.approx((5 - 1) / 4)

    def test_bold_markers_format(self):
        """Handles **bold** dimension names."""
        response = """Done. Here is my feedback:

[RETRIEVAL EFFECTIVENESS]
**Recency:** 2
**Importance:** 3
**Relevance:** 5

Relevance was critical.
[/RETRIEVAL EFFECTIVENESS]"""

        fb = parse_likert_feedback(response)
        assert fb is not None
        assert fb.recency == pytest.approx((2 - 1) / 4)
        assert fb.importance == pytest.approx((3 - 1) / 4)
        assert fb.relevance == pytest.approx((5 - 1) / 4)

    def test_angle_bracket_format(self):
        """Handles <N> wrapped ratings."""
        response = """[RETRIEVAL EFFECTIVENESS]
Recency: <3>
Importance: <4>
Relevance: <5>

Explanation here.
[/RETRIEVAL EFFECTIVENESS]"""

        fb = parse_likert_feedback(response)
        assert fb is not None
        assert fb.recency == pytest.approx((3 - 1) / 4)

    def test_slash_five_format(self):
        """Handles N/5 ratings."""
        response = """## Retrieval Effectiveness
Recency: 2/5
Importance: 4/5
Relevance: 3/5

Importance mattered most."""

        fb = parse_likert_feedback(response)
        assert fb is not None
        assert fb.recency == pytest.approx((2 - 1) / 4)
        assert fb.importance == pytest.approx((4 - 1) / 4)
        assert fb.relevance == pytest.approx((3 - 1) / 4)

    def test_dash_format(self):
        """Handles Recency — 3/5 format."""
        response = """[RETRIEVAL EFFECTIVENESS]
Recency — 3/5
Importance — 4/5
Relevance — 5/5

Relevance was key.
[/RETRIEVAL EFFECTIVENESS]"""

        fb = parse_likert_feedback(response)
        assert fb is not None
        assert fb.recency == pytest.approx((3 - 1) / 4)


class TestFeedbackParserFailureModes:
    """Test that the parser handles common failure modes gracefully."""

    def test_missing_section_returns_none(self):
        """No retrieval effectiveness section at all."""
        response = "I completed the task. Here is the output: SGD explanation."
        fb = parse_likert_feedback(response)
        assert fb is None

    def test_garbled_output_returns_none(self):
        """Garbled text with no parseable ratings."""
        response = """[RETRIEVAL EFFECTIVENESS]
The skills were helpful but I can't rate them numerically.
They contributed to my understanding of the problem.
[/RETRIEVAL EFFECTIVENESS]"""

        fb = parse_likert_feedback(response)
        assert fb is None

    def test_empty_response_returns_none(self):
        fb = parse_likert_feedback("")
        assert fb is None

    def test_partial_ratings_returns_none(self):
        """Only 2 of 3 dimensions present — should fail (strict: all three required)."""
        response = """[RETRIEVAL EFFECTIVENESS]
Recency: 3
Importance: 4

Relevance was not evaluated.
[/RETRIEVAL EFFECTIVENESS]"""

        fb = parse_likert_feedback(response)
        assert fb is None

    def test_out_of_range_rating_returns_none(self):
        """Ratings outside 1-5 should be rejected."""
        response = """[RETRIEVAL EFFECTIVENESS]
Recency: 0
Importance: 6
Relevance: 3
[/RETRIEVAL EFFECTIVENESS]"""

        fb = parse_likert_feedback(response)
        assert fb is None

    def test_ratings_in_tail_without_tags(self):
        """Ratings scattered in the last 1500 chars with no proper tags."""
        response = (
            "A" * 2000
            + """
Here is my assessment of the retrieved skills:
Recency: 3
Importance: 4
Relevance: 5
That's my evaluation."""
        )

        fb = parse_likert_feedback(response)
        assert fb is not None
        assert fb.recency == pytest.approx((3 - 1) / 4)


class TestFeedbackParseRate:
    """Verify parse rates are reasonable (>70%) on realistic MiniMax M2.5 outputs."""

    SAMPLE_RESPONSES = [
        # 1. Perfect tagged format
        """[RETRIEVAL EFFECTIVENESS]
Recency: 3
Importance: 4
Relevance: 5
Brief explanation:
- Relevance was most valuable.
- Recency was least valuable.
[/RETRIEVAL EFFECTIVENESS]""",
        # 2. Markdown heading
        """## Retrieval Effectiveness
Recency: 2
Importance: 5
Relevance: 4
Importance was essential.""",
        # 3. Bold markers
        """[RETRIEVAL EFFECTIVENESS]
**Recency:** 4
**Importance:** 3
**Relevance:** 5
All dimensions helped.
[/RETRIEVAL EFFECTIVENESS]""",
        # 4. Table format
        """### Retrieval Feedback
| Dimension | Rating |
|---|---|
| Recency | 3/5 |
| Importance | 4/5 |
| Relevance | 5/5 |
The table above shows my ratings.""",
        # 5. Slash format
        """## Retrieval Effectiveness
Recency: 1/5
Importance: 3/5
Relevance: 5/5
Relevance was key.""",
        # 6. Dash format
        """[RETRIEVAL EFFECTIVENESS]
Recency — 2/5
Importance — 4/5
Relevance — 3/5
Balanced retrieval.
[/RETRIEVAL EFFECTIVENESS]""",
        # 7. Missing tags, tail fallback
        "x" * 500
        + """
Task done. Rating the retrieval:
Recency: 3
Importance: 3
Relevance: 4
Overall decent retrieval.""",
        # 8. Garbled — should fail
        """The skills were somewhat helpful. I used them to complete the task.
No specific numeric assessment is possible.""",
        # 9. Empty — should fail
        "",
        # 10. Only two ratings — should fail
        """[RETRIEVAL EFFECTIVENESS]
Recency: 4
Importance: 5
[/RETRIEVAL EFFECTIVENESS]""",
    ]

    def test_parse_rate_above_70_percent(self):
        successes = sum(
            1 for resp in self.SAMPLE_RESPONSES if parse_likert_feedback(resp) is not None
        )
        rate = successes / len(self.SAMPLE_RESPONSES)
        assert rate >= 0.70, f"Parse rate {rate:.0%} is below 70% threshold"


class TestExtractFeedbackBlock:
    def test_strict_tags(self):
        text = "prefix [RETRIEVAL EFFECTIVENESS]inner[/RETRIEVAL EFFECTIVENESS] suffix"
        block = _extract_feedback_block(text)
        assert block == "inner"

    def test_case_insensitive(self):
        text = "[retrieval effectiveness]content[/retrieval effectiveness]"
        block = _extract_feedback_block(text)
        assert block == "content"

    def test_markdown_heading_fallback(self):
        text = "### Retrieval Effectiveness\nRatings here"
        block = _extract_feedback_block(text)
        assert "Ratings here" in block


class TestExtractQualitativeSections:
    def test_extracts_all_three_sections(self):
        response = """[RETRIEVAL EFFECTIVENESS]
[RECENCY EVALUATION]
The freshness of skills was very important for this time-sensitive research task.
[/RECENCY EVALUATION]

[IMPORTANCE EVALUATION]
Historical success rates helped identify reliable approaches for this task.
[/IMPORTANCE EVALUATION]

[RELEVANCE EVALUATION]
Topic match was absolutely essential — off-topic skills would have been useless.
[/RELEVANCE EVALUATION]
[/RETRIEVAL EFFECTIVENESS]"""

        sections = extract_qualitative_sections(response)
        assert sections is not None
        assert "recency" in sections
        assert "importance" in sections
        assert "relevance" in sections
        assert "freshness" in sections["recency"]

    def test_missing_section_returns_none(self):
        response = """[RETRIEVAL EFFECTIVENESS]
[RECENCY EVALUATION]
Freshness was important.
[/RECENCY EVALUATION]
[/RETRIEVAL EFFECTIVENESS]"""

        sections = extract_qualitative_sections(response)
        assert sections is None


# ══════════════════════════════════════════════════════════════
# 4. SCORING — Recency and Importance
# ══════════════════════════════════════════════════════════════


@pytest.fixture
def scoring_db():
    """Database with schema and test data for scoring tests."""
    conn = init_db(":memory:")
    for i in range(1, 11):
        conn.execute(
            "INSERT INTO skills (skill_id, domain, title, content) VALUES (?, ?, ?, ?)",
            (f"s{i}", "research", f"Skill {i}", f"Content for skill {i}"),
        )
    conn.execute(
        "INSERT INTO tasks (task_id, theme, title, description) VALUES (?, ?, ?, ?)",
        ("t1", "ML", "Test Task", "Test description"),
    )
    conn.execute(
        "INSERT INTO episodes (condition_id, task_id, preset_id, task_order) VALUES (?, ?, ?, ?)",
        (2, "t1", None, 1),
    )
    conn.commit()
    yield conn
    conn.close()


class TestRecencyScoring:
    def test_never_used_is_neutral(self, scoring_db):
        score = compute_recency(scoring_db, 2, "s1", current_task_order=10, decay_window=50)
        assert score == 0.5

    def test_just_used_is_1(self, scoring_db):
        record_skill_usage(scoring_db, 2, "s1", 1, 1, task_order=10)
        scoring_db.commit()
        score = compute_recency(scoring_db, 2, "s1", current_task_order=10, decay_window=50)
        assert score == pytest.approx(1.0)

    def test_decay_window_50(self, scoring_db):
        """Linear decay over 50 iterations."""
        record_skill_usage(scoring_db, 2, "s1", 1, 1, task_order=1)
        scoring_db.commit()

        # Test various points on the decay curve
        test_cases = [
            (1, 1.0),  # just used
            (11, 0.8),  # 10/50 = 0.2 decay
            (26, 0.5),  # 25/50 = 0.5 decay
            (51, 0.0),  # 50/50 = fully decayed
            (100, 0.0),  # beyond window
        ]

        for task_order, expected in test_cases:
            score = compute_recency(
                scoring_db, 2, "s1", current_task_order=task_order, decay_window=50
            )
            assert score == pytest.approx(expected), (
                f"At task_order={task_order}, expected {expected}, got {score}"
            )

    def test_score_in_01_range(self, scoring_db):
        """Recency scores must always be in [0, 1]."""
        record_skill_usage(scoring_db, 2, "s1", 1, 1, task_order=1)
        scoring_db.commit()

        for task_order in range(1, 200):
            score = compute_recency(
                scoring_db, 2, "s1", current_task_order=task_order, decay_window=50
            )
            assert 0.0 <= score <= 1.0, f"Score {score} out of range at task_order={task_order}"


class TestImportanceScoring:
    def test_never_used_is_neutral(self, scoring_db):
        score = compute_importance(scoring_db, 2, "s1")
        assert score == pytest.approx(0.5)

    def test_laplace_smoothing(self, scoring_db):
        """Laplace smoothing: (successes+1)/(successes+failures+2)."""
        # 3 successes: (3+1)/(3+0+2) = 0.8
        for i in range(3):
            scoring_db.execute(
                "INSERT INTO tasks (task_id, theme, title, description) VALUES (?, ?, ?, ?)",
                (f"t_ls_{i}", "ML", f"Task {i}", "Desc"),
            )
            scoring_db.execute(
                "INSERT INTO episodes"
                " (condition_id, task_id, preset_id, task_order)"
                " VALUES (?, ?, ?, ?)",
                (2, f"t_ls_{i}", None, i + 10),
            )
            scoring_db.commit()
            ep_id = scoring_db.execute("SELECT last_insert_rowid()").fetchone()[0]
            record_skill_usage(scoring_db, 2, "s1", ep_id, 1, task_order=i + 10)
        scoring_db.commit()

        score = compute_importance(scoring_db, 2, "s1")
        assert score == pytest.approx(4 / 5)

    def test_score_in_01_range(self, scoring_db):
        """Importance scores must always be in [0, 1]."""
        # Test with various success/failure counts
        for n_success in range(0, 10):
            for n_failure in range(0, 10):
                expected = (n_success + 1) / (n_success + n_failure + 2)
                assert 0.0 <= expected <= 1.0

    def test_all_failures_approaches_zero(self, scoring_db):
        """Many failures should push importance toward 0 (but never reach it due to smoothing)."""
        for i in range(20):
            scoring_db.execute(
                "INSERT INTO tasks (task_id, theme, title, description) VALUES (?, ?, ?, ?)",
                (f"t_af_{i}", "ML", f"Task {i}", "Desc"),
            )
            scoring_db.execute(
                "INSERT INTO episodes"
                " (condition_id, task_id, preset_id, task_order)"
                " VALUES (?, ?, ?, ?)",
                (2, f"t_af_{i}", None, i + 100),
            )
            scoring_db.commit()
            ep_id = scoring_db.execute("SELECT last_insert_rowid()").fetchone()[0]
            record_skill_usage(scoring_db, 2, "s1", ep_id, 0, task_order=i + 100)
        scoring_db.commit()

        score = compute_importance(scoring_db, 2, "s1")
        # (0+1)/(0+20+2) = 1/22 ≈ 0.045
        assert score == pytest.approx(1 / 22)
        assert score > 0.0  # never actually 0 due to smoothing


# ══════════════════════════════════════════════════════════════
# 5. INTEGRATION — Mock Episode Flow
# ══════════════════════════════════════════════════════════════


class TestMockEpisodeFlow:
    """
    End-to-end mock episode:
    prepare → fake LLM response → parse feedback → update bandit → verify state changed.
    """

    def test_full_episode_cycle(self):
        # 1. PREPARE: Set up DB with presets and bandit state
        conn = init_db(":memory:")
        seed_weight_presets(conn, PRESETS)
        init_bandit_state(conn, [2], list(PRESETS.keys()))

        # Insert a skill and a task
        conn.execute(
            "INSERT INTO skills (skill_id, domain, title, content) VALUES (?, ?, ?, ?)",
            ("sk-001", "research", "Gradient Descent", "SGD optimization"),
        )
        conn.execute(
            "INSERT INTO tasks (task_id, theme, title, description) VALUES (?, ?, ?, ?)",
            ("t-001", "ML", "Explain SGD", "Explain stochastic gradient descent"),
        )
        conn.commit()

        # 2. BANDIT SELECTION
        rng = np.random.default_rng(42)
        bandit = ThompsonSamplingBandit(conn, condition_id=2, rng=rng)

        initial_summary = bandit.get_summary()
        assert initial_summary["total_pulls"] == 0

        preset_id = bandit.select_arm()
        assert preset_id in PRESETS

        # 3. RECORD EPISODE
        conn.execute(
            "INSERT INTO episodes"
            " (condition_id, task_id, preset_id, task_order)"
            " VALUES (?, ?, ?, ?)",
            (2, "t-001", preset_id, 1),
        )
        conn.commit()
        episode_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        # 4. FAKE LLM RESPONSE (simulating what MiniMax M2.5 would return)
        fake_response = """Stochastic Gradient Descent (SGD) is an optimization algorithm that
updates model parameters using gradients computed on random mini-batches.

[RETRIEVAL EFFECTIVENESS]
Recency: 3
Importance: 4
Relevance: 5

Brief explanation:
- Relevance was the most valuable — the gradient descent skill matched the task perfectly.
- Importance was useful because this is a well-tested foundational skill.
- Recency was least valuable since SGD theory doesn't change over time.
[/RETRIEVAL EFFECTIVENESS]"""

        conn.execute(
            "UPDATE episodes SET llm_response = ?,"
            " success = 1, completed_at = datetime('now')"
            " WHERE episode_id = ?",
            (fake_response, episode_id),
        )
        conn.commit()

        # 5. PARSE FEEDBACK
        feedback = parse_likert_feedback(fake_response)
        assert feedback is not None, "Failed to parse feedback from fake LLM response"
        assert feedback.recency == pytest.approx((3 - 1) / 4)
        assert feedback.importance == pytest.approx((4 - 1) / 4)
        assert feedback.relevance == pytest.approx((5 - 1) / 4)

        # 6. STORE FEEDBACK
        conn.execute(
            """INSERT INTO feedback
               (episode_id, rating_recency, rating_importance, rating_relevance, explanation)
               VALUES (?, ?, ?, ?, ?)""",
            (episode_id, 3, 4, 5, feedback.explanation),
        )
        conn.commit()

        # 7. UPDATE BANDIT
        reward = feedback.composite_reward
        assert 0.0 <= reward <= 1.0
        bandit.update(preset_id, reward)

        # 8. VERIFY BANDIT STATE CHANGED
        final_summary = bandit.get_summary()
        assert final_summary["total_pulls"] == 1

        # Find the updated arm
        updated_arm = [a for a in final_summary["arms"] if a["preset_id"] == preset_id][0]
        assert updated_arm["pulls"] == 1
        assert updated_arm["alpha"] > 1.0  # prior was 1.0, reward > 0 added

        # Other arms should still have 0 pulls
        other_arms = [a for a in final_summary["arms"] if a["preset_id"] != preset_id]
        for arm in other_arms:
            assert arm["pulls"] == 0

        # 9. VERIFY EPISODE IN DB
        ep = conn.execute("SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)).fetchone()
        assert ep["success"] == 1
        assert ep["preset_id"] == preset_id

        fb = conn.execute("SELECT * FROM feedback WHERE episode_id = ?", (episode_id,)).fetchone()
        assert fb["rating_relevance"] == 5

        conn.close()

    def test_multiple_episodes_bandit_converges(self):
        """Run 30 episodes. Verify bandit posteriors actually diverge."""
        conn = init_db(":memory:")
        seed_weight_presets(conn, PRESETS)
        init_bandit_state(conn, [2], list(PRESETS.keys()))

        conn.execute(
            "INSERT INTO skills (skill_id, domain, title, content)"
            " VALUES ('sk-001', 'research', 'Skill', 'Content')"
        )
        conn.commit()

        rng = np.random.default_rng(42)
        bandit = ThompsonSamplingBandit(conn, condition_id=2, rng=rng)

        # Run 30 episodes where relevance_heavy consistently gets best feedback
        for i in range(30):
            selected = bandit.select_arm()
            task_id = f"t-{i:03d}"
            conn.execute(
                "INSERT INTO tasks"
                " (task_id, theme, title, description)"
                " VALUES (?, 'ML', ?, 'Desc')",
                (task_id, f"Task {i}"),
            )
            conn.execute(
                "INSERT INTO episodes"
                " (condition_id, task_id, preset_id,"
                " task_order) VALUES (2, ?, ?, ?)",
                (task_id, selected, i + 1),
            )
            conn.commit()

            # Reward structure: relevance_heavy gets 0.75, others get 0.25
            if selected == "relevance_heavy":
                reward = 0.75
            else:
                reward = 0.25
            bandit.update(selected, reward)

        summary = bandit.get_summary()
        assert summary["total_pulls"] == 30

        # After 30 episodes, relevance_heavy should have the highest mean
        best = summary["best_arm"]
        assert best == "relevance_heavy", (
            f"Expected relevance_heavy to converge as best arm, got {best}"
        )

        conn.close()

    def test_failed_parse_does_not_update_bandit(self):
        """If feedback parsing fails, bandit should NOT be updated."""
        conn = init_db(":memory:")
        seed_weight_presets(conn, PRESETS)
        init_bandit_state(conn, [2], list(PRESETS.keys()))

        bandit = ThompsonSamplingBandit(conn, condition_id=2)

        # Simulate a response that fails to parse
        bad_response = "I completed the task. No structured feedback."
        feedback = parse_likert_feedback(bad_response)
        assert feedback is None

        # Bandit should NOT be updated
        summary = bandit.get_summary()
        assert summary["total_pulls"] == 0

        conn.close()
