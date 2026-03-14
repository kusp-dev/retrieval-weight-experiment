"""Tests for dual rate control (sliding window + step budget)."""

import time
from unittest.mock import patch

import pytest

from src.experiment.rate_control import (
    SYNTHESIS_NUDGE,
    RateControlResult,
    SlidingWindowLimiter,
    StepBudget,
)

# ── Sliding Window Limiter ──


class TestSlidingWindowLimiter:
    def test_allows_under_limit(self):
        limiter = SlidingWindowLimiter(max_per_window=10, window_seconds=3600)
        result = limiter.check()
        assert result.action == "go"
        assert result.wait_seconds == 0

    def test_throttles_at_limit(self):
        limiter = SlidingWindowLimiter(max_per_window=3, window_seconds=3600)

        # Record 3 iterations
        for _ in range(3):
            limiter.record()

        result = limiter.check()
        assert result.action == "throttle"
        assert result.wait_seconds > 0

    def test_current_count_tracks_recordings(self):
        limiter = SlidingWindowLimiter(max_per_window=50, window_seconds=3600)
        assert limiter.current_count == 0

        limiter.record()
        limiter.record()
        assert limiter.current_count == 2

    def test_window_expiry_clears_old_entries(self):
        """Entries older than the window should be evicted."""
        limiter = SlidingWindowLimiter(max_per_window=3, window_seconds=1)

        # Record 3 iterations
        for _ in range(3):
            limiter.record()

        # Should be throttled
        assert limiter.check().action == "throttle"

        # Mock time to advance past the window
        with patch("src.experiment.rate_control.time") as mock_time:
            mock_time.time.return_value = time.time() + 2
            result = limiter.check()
            assert result.action == "go"

    def test_reason_contains_count(self):
        limiter = SlidingWindowLimiter(max_per_window=50, window_seconds=3600)
        limiter.record()
        result = limiter.check()
        assert "1/50" in result.reason

    def test_throttle_reason_contains_wait_time(self):
        limiter = SlidingWindowLimiter(max_per_window=1, window_seconds=3600)
        limiter.record()
        result = limiter.check()
        assert "wait" in result.reason


# ── Step Budget ──


class TestStepBudget:
    def test_default_budget(self):
        budget = StepBudget()
        assert budget.max_steps == 35
        assert budget.current_step == 0
        assert budget.remaining == 35

    def test_step_increments(self):
        budget = StepBudget(max_steps=10)
        exhausted = budget.step()
        assert not exhausted
        assert budget.current_step == 1
        assert budget.remaining == 9

    def test_exhaustion(self):
        budget = StepBudget(max_steps=3)
        budget.step()  # 1
        budget.step()  # 2
        exhausted = budget.step()  # 3 = max
        assert exhausted

    def test_reset(self):
        budget = StepBudget(max_steps=10)
        for _ in range(5):
            budget.step()
        budget.reset()
        assert budget.current_step == 0
        assert budget.remaining == 10
        assert not budget.nudge_injected

    def test_fraction_used(self):
        budget = StepBudget(max_steps=10)
        for _ in range(5):
            budget.step()
        assert budget.fraction_used == pytest.approx(0.5)


class TestStepBudgetNudge:
    def test_nudge_at_threshold(self):
        """Nudge should fire at 60% of budget with no output."""
        budget = StepBudget(max_steps=10, nudge_threshold=0.6)

        # Steps 1-5: no nudge
        for _ in range(5):
            budget.step()
            assert not budget.should_nudge(has_output=False)

        # Step 6 (60% of 10): should nudge
        budget.step()
        assert budget.should_nudge(has_output=False)

    def test_nudge_only_fires_once(self):
        budget = StepBudget(max_steps=10, nudge_threshold=0.6)

        # Advance to threshold
        for _ in range(6):
            budget.step()

        assert budget.should_nudge(has_output=False)  # first time: True
        assert not budget.should_nudge(has_output=False)  # second time: False

    def test_no_nudge_if_output_exists(self):
        """Nudge should NOT fire if the agent already has meaningful output."""
        budget = StepBudget(max_steps=10, nudge_threshold=0.6)
        for _ in range(6):
            budget.step()

        assert not budget.should_nudge(has_output=True)

    def test_nudge_reset_clears_flag(self):
        budget = StepBudget(max_steps=10, nudge_threshold=0.6)
        for _ in range(6):
            budget.step()
        budget.should_nudge(has_output=False)  # fires
        assert budget.nudge_injected

        budget.reset()
        assert not budget.nudge_injected


class TestSynthesisNudge:
    def test_nudge_text_exists(self):
        assert len(SYNTHESIS_NUDGE) > 0
        assert "60%" in SYNTHESIS_NUDGE
        assert "synthesiz" in SYNTHESIS_NUDGE.lower()


class TestRateControlResult:
    def test_go_result(self):
        result = RateControlResult(action="go", wait_seconds=0, reason="under limit")
        assert result.action == "go"
        assert result.wait_seconds == 0

    def test_throttle_result(self):
        result = RateControlResult(action="throttle", wait_seconds=60, reason="over limit")
        assert result.action == "throttle"
        assert result.wait_seconds == 60
