"""
Dual rate control adapted from TIER 0 production system.

Layer 1: Iteration-level sliding window rate limiter
  - Counts completed iterations (not individual API steps)
  - MiniMax budget: 1,000 prompts per 5-hour window
  - With 4 conditions running sequentially: 50 iterations/hour per condition

Layer 2: Per-iteration step budget (max_steps=35)
  - Prevents runaway iterations from eating the hourly budget
  - Metacognitive guardrail at 60% budget: inject synthesis nudge

Design ref: SYSTEM_DESIGN.md §9.1-9.2
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class RateControlResult:
    """Result of a rate limit check."""

    action: str  # "go" or "throttle"
    wait_seconds: int  # 0 if action is "go"
    reason: str


class SlidingWindowLimiter:
    """Layer 1: Iteration-level rate limiter.

    Counts completed iterations in a sliding time window.
    """

    def __init__(
        self,
        max_per_window: int = 50,
        window_seconds: int = 3600,
    ):
        self.max_per_window = max_per_window
        self.window_seconds = window_seconds
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()

    def check(self) -> RateControlResult:
        """Check if we can proceed or need to wait. Thread-safe."""
        with self._lock:
            now = time.time()

            # Evict expired timestamps
            while self._timestamps and (now - self._timestamps[0]) > self.window_seconds:
                self._timestamps.popleft()

            count = len(self._timestamps)

            if count >= self.max_per_window:
                oldest = self._timestamps[0]
                wait = int(self.window_seconds - (now - oldest)) + 1
                return RateControlResult(
                    action="throttle",
                    wait_seconds=wait,
                    reason=f"Rate: {count}/{self.max_per_window}/window, wait {wait}s",
                )

            return RateControlResult(
                action="go",
                wait_seconds=0,
                reason=f"Rate: {count}/{self.max_per_window}/window",
            )

    def record(self) -> None:
        """Record a completed iteration. Thread-safe."""
        with self._lock:
            self._timestamps.append(time.time())

    @property
    def current_count(self) -> int:
        """Current count of iterations in the window."""
        with self._lock:
            now = time.time()
            while self._timestamps and (now - self._timestamps[0]) > self.window_seconds:
                self._timestamps.popleft()
            return len(self._timestamps)


SYNTHESIS_NUDGE = (
    "You've used 60% of your step budget. Focus on synthesizing what you have "
    "into a complete response rather than continuing to research. Use the skills "
    "and information already retrieved."
)


@dataclass
class StepBudget:
    """Layer 2: Per-iteration step budget with metacognitive guardrail.

    Tracks step count within a single task iteration and injects a
    synthesis nudge at 60% of the budget.
    """

    max_steps: int = 35
    nudge_threshold: float = 0.6  # fraction of budget
    current_step: int = field(default=0, init=False)
    nudge_injected: bool = field(default=False, init=False)

    def reset(self) -> None:
        """Reset for a new iteration."""
        self.current_step = 0
        self.nudge_injected = False

    def step(self) -> bool:
        """Increment step count. Returns True if budget exhausted."""
        self.current_step += 1
        return self.current_step >= self.max_steps

    def should_nudge(self, has_output: bool) -> bool:
        """Check if synthesis nudge should be injected.

        Triggers at 60% budget consumed AND no meaningful output yet.
        Only fires once per iteration.
        """
        if self.nudge_injected:
            return False

        threshold_step = int(self.max_steps * self.nudge_threshold)
        if self.current_step >= threshold_step and not has_output:
            self.nudge_injected = True
            return True

        return False

    @property
    def remaining(self) -> int:
        return max(0, self.max_steps - self.current_step)

    @property
    def fraction_used(self) -> float:
        return self.current_step / self.max_steps
