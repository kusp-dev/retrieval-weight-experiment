"""Tests for PID step budget controller."""

import pytest

from src.experiment.pid_controller import PIDStepController


class TestInitialization:
    def test_default_parameters(self):
        """Default construction should set expected values."""
        ctrl = PIDStepController()
        assert ctrl.setpoint == 5.0
        assert ctrl.kp == 1.0
        assert ctrl.ki == 0.1
        assert ctrl.kd == 0.5
        assert ctrl.window == 20
        assert ctrl.min_steps == 1
        assert ctrl.max_steps == 35

    def test_custom_parameters(self):
        """All parameters should be overridable."""
        ctrl = PIDStepController(
            setpoint=10.0,
            kp=2.0,
            ki=0.5,
            kd=1.0,
            window=50,
            min_steps=3,
            max_steps=100,
        )
        assert ctrl.setpoint == 10.0
        assert ctrl.kp == 2.0
        assert ctrl.ki == 0.5
        assert ctrl.kd == 1.0
        assert ctrl.window == 50
        assert ctrl.min_steps == 3
        assert ctrl.max_steps == 100

    def test_initial_state_is_clean(self):
        """Fresh controller should have empty history and no accumulated state."""
        ctrl = PIDStepController()
        state = ctrl.get_state()
        assert state["history_len"] == 0
        assert state["integral"] == 0.0
        assert state["prev_error"] is None
        assert state["process_variable"] == 0.0

    def test_anti_windup_limit_computed_from_output_range(self):
        """Integral limit should be (max_steps - min_steps) / ki."""
        ctrl = PIDStepController(ki=0.2, min_steps=5, max_steps=25)
        expected = (25 - 5) / 0.2
        assert ctrl._integral_limit == pytest.approx(expected)

    def test_anti_windup_limit_zero_ki(self):
        """With ki=0 the anti-windup limit uses epsilon to avoid division by zero."""
        ctrl = PIDStepController(ki=0.0, min_steps=1, max_steps=35)
        assert ctrl._integral_limit == pytest.approx((35 - 1) / 1e-9)

    def test_history_deque_has_correct_maxlen(self):
        """Internal history deque should enforce the window size."""
        ctrl = PIDStepController(window=10)
        assert ctrl._history.maxlen == 10


class TestProportionalComponent:
    def test_error_above_setpoint_lowers_budget(self):
        """When steps consistently exceed setpoint, budget should decrease."""
        ctrl = PIDStepController(setpoint=5.0, kp=1.0, ki=0.0, kd=0.0)
        # Feed step count of 15 — error = 15 - 5 = 10, P = 10
        # raw = 35 - 10 = 25
        result = ctrl.update(15)
        assert result == 25

    def test_error_below_setpoint_raises_budget(self):
        """When steps are below setpoint, budget should increase (toward max)."""
        ctrl = PIDStepController(setpoint=10.0, kp=1.0, ki=0.0, kd=0.0)
        # Feed step count of 3 — error = 3 - 10 = -7, P = -7
        # raw = 35 - (-7) = 42, clamped to 35
        result = ctrl.update(3)
        assert result == 35

    def test_zero_error_returns_max_steps(self):
        """When process variable equals setpoint, output should be max_steps."""
        ctrl = PIDStepController(setpoint=5.0, kp=1.0, ki=0.0, kd=0.0)
        result = ctrl.update(5)
        assert result == 35

    def test_large_kp_amplifies_response(self):
        """Larger kp should produce a stronger correction."""
        ctrl_low = PIDStepController(setpoint=5.0, kp=0.5, ki=0.0, kd=0.0)
        ctrl_high = PIDStepController(setpoint=5.0, kp=3.0, ki=0.0, kd=0.0)
        result_low = ctrl_low.update(10)
        result_high = ctrl_high.update(10)
        # Both should reduce budget, but high kp reduces more
        assert result_high < result_low


class TestIntegralComponent:
    def test_integral_accumulates_over_updates(self):
        """Repeated positive error should accumulate in the integral."""
        ctrl = PIDStepController(setpoint=5.0, kp=0.0, ki=0.5, kd=0.0)
        # Three updates at step_count=10 → error=5 each time
        ctrl.update(10)
        ctrl.update(10)
        result = ctrl.update(10)
        state = ctrl.get_state()
        # integral = 5 + 5 + 5 = 15, I = 0.5 * 15 = 7.5
        # raw = 35 - 7.5 = 27.5 → round → 28
        assert state["integral"] == pytest.approx(15.0)
        assert result == 28

    def test_integral_anti_windup_clamp(self):
        """Integral should be clamped to prevent runaway accumulation."""
        ctrl = PIDStepController(
            setpoint=5.0,
            kp=0.0,
            ki=0.5,
            kd=0.0,
            min_steps=1,
            max_steps=35,
        )
        # integral_limit = (35 - 1) / 0.5 = 68
        # Push 100 updates with error = 30 each → integral would be 3000 without clamp
        for _ in range(100):
            ctrl.update(35)
        state = ctrl.get_state()
        assert state["integral"] == pytest.approx(68.0)

    def test_negative_integral_accumulation(self):
        """Negative errors should decrease the integral."""
        ctrl = PIDStepController(setpoint=10.0, kp=0.0, ki=0.5, kd=0.0)
        # step_count=3 → error = 3 - 10 = -7 each time
        ctrl.update(3)
        ctrl.update(3)
        state = ctrl.get_state()
        assert state["integral"] == pytest.approx(-14.0)

    def test_integral_cancellation(self):
        """Positive then negative errors should partially cancel integral."""
        ctrl = PIDStepController(setpoint=5.0, kp=0.0, ki=1.0, kd=0.0)
        ctrl.update(10)  # error = +5, integral = 5
        ctrl.update(0)  # error = (10+0)/2 - 5 = 0, integral = 5 + 0 = 5
        state = ctrl.get_state()
        # After second update: pv = (10+0)/2 = 5.0, error = 0
        assert state["integral"] == pytest.approx(5.0)


class TestDerivativeComponent:
    def test_first_update_has_no_derivative(self):
        """On the first call, D term should be zero (no previous error)."""
        ctrl = PIDStepController(setpoint=5.0, kp=0.0, ki=0.0, kd=1.0)
        # error = 10 - 5 = 5, but D = 0 since no prev_error
        # raw = 35 - 0 = 35
        result = ctrl.update(10)
        assert result == 35

    def test_derivative_responds_to_error_change(self):
        """D term should capture the rate of change of error."""
        ctrl = PIDStepController(setpoint=5.0, kp=0.0, ki=0.0, kd=1.0)
        ctrl.update(10)  # error = 5, D = 0 (first call)
        # Second update: pv = (10+20)/2 = 15, error = 10
        # D = 1.0 * (10 - 5) = 5, raw = 35 - 5 = 30
        result = ctrl.update(20)
        assert result == 30

    def test_derivative_zero_when_error_constant(self):
        """If error doesn't change between calls, D term should be zero."""
        ctrl = PIDStepController(setpoint=0.0, kp=0.0, ki=0.0, kd=1.0, window=1)
        # window=1 means pv is just the latest value
        ctrl.update(10)  # error = 10, D = 0
        result = ctrl.update(10)  # error = 10, D = 1.0 * (10-10) = 0
        # raw = 35 - 0 = 35
        assert result == 35


class TestOutputClamping:
    def test_output_never_below_min_steps(self):
        """Extremely high step counts should not push output below min_steps."""
        ctrl = PIDStepController(setpoint=5.0, kp=5.0, ki=0.0, kd=0.0, min_steps=3)
        # step_count=100 → error=95, P=475, raw=35-475=-440 → clamped to 3
        result = ctrl.update(100)
        assert result == 3

    def test_output_never_above_max_steps(self):
        """Extremely low step counts should not push output above max_steps."""
        ctrl = PIDStepController(setpoint=50.0, kp=5.0, ki=0.0, kd=0.0, max_steps=35)
        # step_count=1 → error=1-50=-49, P=-245, raw=35-(-245)=280 → clamped to 35
        result = ctrl.update(1)
        assert result == 35

    def test_output_is_integer(self):
        """update() should always return an int."""
        ctrl = PIDStepController()
        result = ctrl.update(7)
        assert isinstance(result, int)

    def test_min_equals_max_always_returns_that_value(self):
        """When min_steps == max_steps, output is locked."""
        ctrl = PIDStepController(min_steps=10, max_steps=10, kp=1.0, ki=0.0, kd=0.0)
        assert ctrl.update(1) == 10
        assert ctrl.update(100) == 10


class TestRollingWindow:
    def test_window_limits_history_length(self):
        """History deque should not exceed the window size."""
        ctrl = PIDStepController(window=5)
        for val in range(20):
            ctrl.update(val)
        assert ctrl.get_state()["history_len"] == 5

    def test_rolling_mean_uses_window(self):
        """Process variable should be the mean of the window, not all observations."""
        ctrl = PIDStepController(window=3, setpoint=5.0, kp=1.0, ki=0.0, kd=0.0)
        # Fill window with high values, then replace with low values
        ctrl.update(100)
        ctrl.update(100)
        ctrl.update(100)
        # Now window = [100, 100, 100], pv = 100
        ctrl.update(1)
        ctrl.update(1)
        ctrl.update(1)
        # Now window = [1, 1, 1], pv = 1
        state = ctrl.get_state()
        assert state["process_variable"] == pytest.approx(1.0)

    def test_single_observation_window(self):
        """With window=1, only the latest observation matters."""
        ctrl = PIDStepController(window=1, setpoint=5.0, kp=1.0, ki=0.0, kd=0.0)
        ctrl.update(100)
        ctrl.update(5)
        # pv = 5, error = 0, raw = 35
        state = ctrl.get_state()
        assert state["process_variable"] == pytest.approx(5.0)


class TestGetState:
    def test_state_structure(self):
        """get_state() should return all expected keys."""
        ctrl = PIDStepController()
        state = ctrl.get_state()
        expected_keys = {
            "setpoint",
            "process_variable",
            "error",
            "integral",
            "prev_error",
            "history_len",
            "kp",
            "ki",
            "kd",
        }
        assert set(state.keys()) == expected_keys

    def test_state_reflects_updates(self):
        """State should reflect the current PID internals after updates."""
        ctrl = PIDStepController(setpoint=5.0, kp=1.0, ki=0.1, kd=0.5)
        ctrl.update(10)
        state = ctrl.get_state()
        assert state["process_variable"] == pytest.approx(10.0)
        assert state["error"] == pytest.approx(5.0)
        assert state["integral"] == pytest.approx(5.0)
        assert state["prev_error"] == pytest.approx(5.0)
        assert state["history_len"] == 1

    def test_state_prev_error_none_before_first_update(self):
        """Before any update, prev_error should be None."""
        ctrl = PIDStepController()
        assert ctrl.get_state()["prev_error"] is None

    def test_state_values_are_rounded(self):
        """Process variable, error, integral should be rounded to 2 decimals."""
        ctrl = PIDStepController(setpoint=3.0)
        ctrl.update(7)
        ctrl.update(2)
        state = ctrl.get_state()
        # All numeric values should have at most 2 decimal places
        assert state["process_variable"] == round(state["process_variable"], 2)
        assert state["error"] == round(state["error"], 2)
        assert state["integral"] == round(state["integral"], 2)


class TestReset:
    def test_reset_clears_history(self):
        """reset() should empty the history deque."""
        ctrl = PIDStepController()
        ctrl.update(10)
        ctrl.update(20)
        ctrl.reset()
        assert ctrl.get_state()["history_len"] == 0

    def test_reset_clears_integral(self):
        """reset() should zero the accumulated integral."""
        ctrl = PIDStepController()
        ctrl.update(10)
        ctrl.update(10)
        ctrl.reset()
        assert ctrl.get_state()["integral"] == 0.0

    def test_reset_clears_prev_error(self):
        """reset() should set prev_error back to None."""
        ctrl = PIDStepController()
        ctrl.update(10)
        ctrl.reset()
        assert ctrl.get_state()["prev_error"] is None

    def test_controller_works_normally_after_reset(self):
        """After reset, controller should behave like a fresh instance."""
        ctrl = PIDStepController(setpoint=5.0, kp=1.0, ki=0.0, kd=0.0)
        # Run it for a while
        for _ in range(50):
            ctrl.update(20)
        ctrl.reset()
        # First update after reset: pv=10, error=5, P=5, raw=35-5=30
        result = ctrl.update(10)
        assert result == 30


class TestNegativeFeedbackBehavior:
    """Tests for the core negative feedback property: too many steps -> lower budget."""

    def test_consistently_high_steps_reduces_budget(self):
        """Repeatedly exceeding the setpoint should progressively tighten the budget."""
        ctrl = PIDStepController(setpoint=5.0, kp=1.0, ki=0.1, kd=0.5)
        results = []
        for _ in range(10):
            results.append(ctrl.update(15))
        # Budget should generally decrease (or at least not increase) over time
        assert results[-1] <= results[0]

    def test_consistently_low_steps_keeps_budget_high(self):
        """Consistently low step counts should keep the budget near max_steps."""
        ctrl = PIDStepController(setpoint=5.0, kp=1.0, ki=0.1, kd=0.5, max_steps=35)
        results = []
        for _ in range(10):
            results.append(ctrl.update(2))
        # Budget should stay at or near max_steps
        assert all(r >= 30 for r in results)

    def test_convergence_to_setpoint_stabilizes_budget(self):
        """When step counts match the setpoint, budget should stabilize near max_steps."""
        ctrl = PIDStepController(setpoint=5.0, kp=1.0, ki=0.0, kd=0.0)
        # Feed exact setpoint values
        results = [ctrl.update(5) for _ in range(10)]
        # All results should be max_steps since error is always 0
        assert all(r == 35 for r in results)


class TestEdgeCases:
    def test_zero_step_count(self):
        """Step count of zero should not cause errors."""
        ctrl = PIDStepController(setpoint=5.0)
        result = ctrl.update(0)
        assert isinstance(result, int)
        assert ctrl.min_steps <= result <= ctrl.max_steps

    def test_very_large_step_count(self):
        """Extremely large step count should clamp output to min_steps."""
        ctrl = PIDStepController(setpoint=5.0, kp=1.0, ki=0.0, kd=0.0, min_steps=1)
        result = ctrl.update(10000)
        assert result == 1

    def test_sign_change_in_error(self):
        """Error crossing zero should be handled smoothly."""
        ctrl = PIDStepController(setpoint=5.0, kp=1.0, ki=0.0, kd=0.0, window=1)
        result_high = ctrl.update(10)  # error = +5
        result_low = ctrl.update(2)  # error = -3
        # High steps -> lower budget; low steps -> higher budget
        assert result_low > result_high

    def test_many_updates_no_crash(self):
        """Controller should be stable over thousands of updates."""
        ctrl = PIDStepController()
        for i in range(5000):
            result = ctrl.update(i % 20)
            assert ctrl.min_steps <= result <= ctrl.max_steps


class TestRealisticScenario:
    """Tests simulating realistic agent step management scenarios."""

    def test_agent_learning_to_be_efficient(self):
        """Simulate an agent that starts slow then gets faster."""
        ctrl = PIDStepController(setpoint=5.0, kp=1.0, ki=0.1, kd=0.5, window=10)
        # Phase 1: Agent uses many steps (learning)
        early_budgets = []
        for _ in range(10):
            early_budgets.append(ctrl.update(15))
        # Phase 2: Agent becomes efficient
        late_budgets = []
        for _ in range(20):
            late_budgets.append(ctrl.update(3))
        # Late budgets should be higher than early budgets on average
        assert sum(late_budgets) / len(late_budgets) > sum(early_budgets) / len(early_budgets)

    def test_bursty_workload(self):
        """Simulate alternating easy and hard tasks."""
        ctrl = PIDStepController(setpoint=5.0, kp=1.0, ki=0.05, kd=0.3, window=10)
        results = []
        for i in range(40):
            steps = 2 if i % 2 == 0 else 12  # alternating easy/hard
            results.append(ctrl.update(steps))
        # All results should be within valid bounds
        assert all(ctrl.min_steps <= r <= ctrl.max_steps for r in results)
        # The budget should hover around a stable range (not wildly oscillate)
        last_10 = results[-10:]
        assert max(last_10) - min(last_10) < 20  # reasonable stability
