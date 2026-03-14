"""
MiniMax M2.5 client implementing the LLMClient protocol.

Uses the Anthropic SDK with MiniMax's Anthropic-compatible endpoint.
Design ref: SYSTEM_DESIGN.md §1 (stack), experiment.yaml (llm section)
"""

import os

import anthropic

from src.experiment.runner import LLMResponse

# Default endpoint for MiniMax's Anthropic-compatible API
DEFAULT_BASE_URL = "https://api.minimax.io/anthropic"
DEFAULT_MODEL = "MiniMax-M2.5"
DEFAULT_TIMEOUT = 300.0
DEFAULT_MAX_RETRIES = 3


class MiniMaxClient:
    """MiniMax M2.5 client that satisfies the LLMClient protocol.

    Uses the Anthropic SDK pointed at MiniMax's compatible endpoint.
    API key is read from MINIMAX_API_KEY environment variable.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        resolved_key = api_key or os.environ.get("MINIMAX_API_KEY")
        if not resolved_key:
            raise ValueError(
                "MiniMax API key required. Pass api_key= or set MINIMAX_API_KEY env var."
            )

        self._client = anthropic.Anthropic(
            base_url=base_url,
            api_key=resolved_key,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._model = model

    def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Send a single prompt and return the response.

        Maps the runner's flat prompt string to Anthropic's messages format.
        """
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )

        # Extract text from content blocks
        text = "".join(block.text for block in response.content if block.type == "text")

        inp = response.usage.input_tokens
        out = response.usage.output_tokens

        return LLMResponse(
            text=text,
            tokens_used=inp + out,
            input_tokens=inp,
            output_tokens=out,
            is_final=(response.stop_reason == "end_turn"),
            model=response.model,
        )
