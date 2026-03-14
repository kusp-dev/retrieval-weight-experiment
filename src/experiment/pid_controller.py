"""
L6: PID step budget controller for adaptive agent step management.

Adjusts the per-task max_steps budget based on rolling performance.
When agents consistently use too many steps, the budget tightens to
force efficiency. When they complete quickly, it relaxes to allow
complexity when needed.

Operates on a different plane than the retrieval weight bandit (L2):
L2 optimizes WHAT the agent retrieves; L6 optimizes HOW LONG it runs.
The compound effect: better retrieval (L2) → fewer steps needed →
L6 tightens budget → more iterations fit in the API window → faster
learning across all layers.

Design ref: ANALYSIS.md §5b (L6 interface), §3f (PID control theory)
"""

from collections import deque


class PIDStepController:
    """PID controller that adapts step budget based on rolling performance.

    Setpoint:  Target average step count (efficient completion).
    Process:   Rolling mean of observed step counts.
    Output:    Recommended max_steps for the next task (clamped).

    The controller INVERTS the error sign: if steps are too high
    (positive error), the output DECREASES max_steps to tighten the
    budget. This is negative feedback — the hallmark of a stable
    control system.
    """

    def __init__(
        self,
        setpoint: float = 5.0,
        kp: float = 1.0,
        ki: float = 0.1,
        kd: float = 0.5,
        window: int = 20,
        min_steps: int = 1,
        max_steps: int = 35,
    ):
        self.setpoint = setpoint
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.window = window
        self.min_steps = min_steps
        self.max_steps = max_steps

        # State
        self._history: deque[int] = deque(maxlen=window)
        self._integral: float = 0.0
        self._prev_error: float | None = None

        # Anti-windup: clamp integral to prevent runaway accumulation.
        # Bounds derived from output range so the I term alone can't
        # exceed the full output span.
        self._integral_limit = (max_steps - min_steps) / max(ki, 1e-9)

    def update(self, step_count: int) -> int:
        """Record a step count observation and return recommended max_steps."""
        self._history.append(step_count)

        # Process variable: rolling mean
        pv = sum(self._history) / len(self._history)
        error = pv - self.setpoint

        # P term
        p = self.kp * error

        # I term with anti-windup clamp
        self._integral += error
        self._integral = max(-self._integral_limit, min(self._integral_limit, self._integral))
        i = self.ki * self._integral

        # D term (zero on first call — no previous error to diff against)
        if self._prev_error is not None:
            d = self.kd * (error - self._prev_error)
        else:
            d = 0.0
        self._prev_error = error

        # PID output: subtract correction from max_steps.
        # Positive error (too many steps) → positive correction → lower budget.
        raw = self.max_steps - (p + i + d)
        return int(max(self.min_steps, min(self.max_steps, round(raw))))

    def get_state(self) -> dict:
        """Return current PID state for logging/monitoring."""
        n = len(self._history)
        pv = sum(self._history) / n if n > 0 else 0.0
        return {
            "setpoint": self.setpoint,
            "process_variable": round(pv, 2),
            "error": round(pv - self.setpoint, 2),
            "integral": round(self._integral, 2),
            "prev_error": round(self._prev_error, 2) if self._prev_error is not None else None,
            "history_len": n,
            "kp": self.kp,
            "ki": self.ki,
            "kd": self.kd,
        }

    def reset(self) -> None:
        """Reset controller state."""
        self._history.clear()
        self._integral = 0.0
        self._prev_error = None
