"""
Langfuse observability for the experiment runner.

Optional: gracefully degrades to no-ops if langfuse is not installed
or credentials are not configured. This keeps the experiment runnable
without Langfuse for reproducibility.

Credentials: Set LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_HOST
in the environment (same .env file as MINIMAX_API_KEY).

Note: langfuse is imported lazily at init time (not module level) to avoid
pydantic v1 compatibility issues with Python 3.14.

API: Uses Langfuse 4.x SDK (start_observation pattern).
"""

import logging
import time

logger = logging.getLogger(__name__)


class ExperimentTracer:
    """Wraps Langfuse client with experiment-specific trace helpers.

    Usage:
        tracer = ExperimentTracer()  # reads keys from env
        trace = tracer.start_episode(condition="full_system", task_id="T001", ...)
        trace.log_llm_call(prompt, response_text, tokens, model)
        trace.log_feedback(parsed=True, reward=0.72, method="likert")
        trace.end(success=True)
    """

    def __init__(self, enabled: bool = True):
        self._client = None
        if not enabled:
            return

        try:
            from langfuse import Langfuse

            client = Langfuse()
            if not client.auth_check():
                logger.info("Langfuse auth check failed — tracing disabled")
                return
            self._client = client
            logger.info("Langfuse tracing enabled")
        except ImportError:
            logger.info("langfuse package not installed — tracing disabled")
        except Exception as e:
            logger.warning(f"Langfuse init failed (tracing disabled): {e}")

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def start_episode(
        self,
        condition_name: str,
        condition_id: int,
        task_id: str,
        task_order: int,
        preset_id: str,
        task_title: str = "",
    ) -> "EpisodeTrace":
        """Start tracing an episode. Returns an EpisodeTrace handle."""
        if not self.enabled or self._client is None:
            return EpisodeTrace(None)

        span = self._client.start_observation(
            name=f"episode:{condition_name}",
            as_type="span",
            metadata={
                "condition_id": condition_id,
                "condition_name": condition_name,
                "task_id": task_id,
                "task_order": task_order,
                "preset_id": preset_id,
                "task_title": task_title,
            },
        )
        return EpisodeTrace(span)

    def flush(self) -> None:
        """Flush pending events to Langfuse."""
        if self._client is not None:
            self._client.flush()

    def shutdown(self) -> None:
        """Flush and shutdown the Langfuse client."""
        if self._client is not None:
            self._client.flush()
            self._client.shutdown()


class EpisodeTrace:
    """Trace handle for a single episode. No-ops if span is None."""

    def __init__(self, span):
        self._span = span
        self._start_time = time.time()

    def log_llm_call(
        self,
        prompt: str,
        response_text: str,
        tokens_used: int,
        model: str,
        step: int = 0,
    ) -> None:
        """Log an LLM generation within this episode."""
        if self._span is None:
            return

        gen = self._span.start_observation(
            name=f"llm_step_{step}",
            as_type="generation",
            input=prompt[-2000:] if len(prompt) > 2000 else prompt,
            output=response_text[-2000:] if len(response_text) > 2000 else response_text,
            model=model,
            usage_details={"total": tokens_used},
            metadata={"step": step, "prompt_length": len(prompt)},
        )
        gen.end()

    def log_feedback(
        self,
        parsed: bool,
        reward: float | None = None,
        method: str = "",
        ratings: tuple[float, float, float] | None = None,
    ) -> None:
        """Log feedback parsing result."""
        if self._span is None:
            return

        metadata = {"parsed": parsed, "method": method}
        if reward is not None:
            metadata["composite_reward"] = reward
        if ratings is not None:
            metadata["recency"] = ratings[0]
            metadata["importance"] = ratings[1]
            metadata["relevance"] = ratings[2]

        child = self._span.start_observation(
            name="feedback_parse",
            as_type="span",
            metadata=metadata,
        )
        child.end()

        if parsed and reward is not None:
            self._span.score(name="composite_reward", value=reward)

    def end(self, success: bool, step_count: int = 0, total_tokens: int = 0) -> None:
        """Mark the episode trace as complete."""
        if self._span is None:
            return

        elapsed = time.time() - self._start_time
        self._span.update(
            metadata={
                "success": success,
                "step_count": step_count,
                "total_tokens": total_tokens,
                "duration_seconds": round(elapsed, 1),
            },
        )
        self._span.end()
