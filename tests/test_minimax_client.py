"""Tests for MiniMax M2.5 client."""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.experiment.runner import LLMResponse
from src.llm.minimax_client import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT,
    MiniMaxClient,
)

# ── Fixtures ──


def _mock_anthropic_response(
    text: str = "Hello world",
    input_tokens: int = 50,
    output_tokens: int = 100,
    stop_reason: str = "end_turn",
    model: str = "MiniMax-M2.5",
):
    """Build a mock Anthropic response object."""
    block = MagicMock()
    block.type = "text"
    block.text = text

    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens

    response = MagicMock()
    response.content = [block]
    response.usage = usage
    response.stop_reason = stop_reason
    response.model = model
    return response


@pytest.fixture
def mock_anthropic():
    """Patch anthropic.Anthropic so no real HTTP calls are made."""
    with patch("src.llm.minimax_client.anthropic.Anthropic") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        mock_instance.messages.create.return_value = _mock_anthropic_response()
        yield mock_cls, mock_instance


# ── Constructor tests ──


class TestMiniMaxClientInit:
    def test_explicit_api_key(self, mock_anthropic):
        mock_cls, _ = mock_anthropic
        MiniMaxClient(api_key="sk-test-123")

        mock_cls.assert_called_once_with(
            base_url=DEFAULT_BASE_URL,
            api_key="sk-test-123",
            timeout=DEFAULT_TIMEOUT,
            max_retries=3,
        )

    def test_env_api_key(self, mock_anthropic):
        mock_cls, _ = mock_anthropic
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "sk-env-456"}):
            MiniMaxClient()

        mock_cls.assert_called_once_with(
            base_url=DEFAULT_BASE_URL,
            api_key="sk-env-456",
            timeout=DEFAULT_TIMEOUT,
            max_retries=3,
        )

    def test_explicit_key_overrides_env(self, mock_anthropic):
        mock_cls, _ = mock_anthropic
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "sk-env-456"}):
            MiniMaxClient(api_key="sk-explicit-789")

        mock_cls.assert_called_once_with(
            base_url=DEFAULT_BASE_URL,
            api_key="sk-explicit-789",
            timeout=DEFAULT_TIMEOUT,
            max_retries=3,
        )

    def test_missing_api_key_raises(self, mock_anthropic):
        with patch.dict(os.environ, {}, clear=True):
            # Remove MINIMAX_API_KEY if it exists
            os.environ.pop("MINIMAX_API_KEY", None)
            with pytest.raises(ValueError, match="MiniMax API key required"):
                MiniMaxClient()

    def test_custom_base_url(self, mock_anthropic):
        mock_cls, _ = mock_anthropic
        MiniMaxClient(api_key="sk-test", base_url="https://custom.api/v1")

        mock_cls.assert_called_once_with(
            base_url="https://custom.api/v1",
            api_key="sk-test",
            timeout=DEFAULT_TIMEOUT,
            max_retries=3,
        )

    def test_custom_model(self, mock_anthropic):
        _, _ = mock_anthropic
        client = MiniMaxClient(api_key="sk-test", model="MiniMax-M3")
        assert client._model == "MiniMax-M3"


# ── Complete method tests ──


class TestMiniMaxClientComplete:
    def test_basic_completion(self, mock_anthropic):
        _, mock_instance = mock_anthropic
        mock_instance.messages.create.return_value = _mock_anthropic_response(
            text="The answer is 42",
            input_tokens=30,
            output_tokens=10,
        )

        client = MiniMaxClient(api_key="sk-test")
        result = client.complete("What is the meaning of life?")

        assert isinstance(result, LLMResponse)
        assert result.text == "The answer is 42"
        assert result.tokens_used == 40  # 30 + 10
        assert result.is_final is True
        assert result.model == "MiniMax-M2.5"

    def test_passes_parameters(self, mock_anthropic):
        _, mock_instance = mock_anthropic

        client = MiniMaxClient(api_key="sk-test")
        client.complete("Test prompt", temperature=0.3, max_tokens=2048)

        mock_instance.messages.create.assert_called_once_with(
            model=DEFAULT_MODEL,
            max_tokens=2048,
            temperature=0.3,
            messages=[{"role": "user", "content": "Test prompt"}],
        )

    def test_default_parameters(self, mock_anthropic):
        _, mock_instance = mock_anthropic

        client = MiniMaxClient(api_key="sk-test")
        client.complete("Test prompt")

        mock_instance.messages.create.assert_called_once_with(
            model=DEFAULT_MODEL,
            max_tokens=4096,
            temperature=0.7,
            messages=[{"role": "user", "content": "Test prompt"}],
        )

    def test_multiple_content_blocks(self, mock_anthropic):
        """Response with multiple text blocks should concatenate them."""
        _, mock_instance = mock_anthropic

        block1 = MagicMock()
        block1.type = "text"
        block1.text = "Part one. "

        block2 = MagicMock()
        block2.type = "text"
        block2.text = "Part two."

        usage = MagicMock()
        usage.input_tokens = 20
        usage.output_tokens = 30

        response = MagicMock()
        response.content = [block1, block2]
        response.usage = usage
        response.stop_reason = "end_turn"
        response.model = "MiniMax-M2.5"

        mock_instance.messages.create.return_value = response

        client = MiniMaxClient(api_key="sk-test")
        result = client.complete("Test")

        assert result.text == "Part one. Part two."
        assert result.tokens_used == 50

    def test_non_text_blocks_skipped(self, mock_anthropic):
        """Non-text content blocks (e.g., tool_use) should be ignored."""
        _, mock_instance = mock_anthropic

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Response text"

        tool_block = MagicMock()
        tool_block.type = "tool_use"

        usage = MagicMock()
        usage.input_tokens = 10
        usage.output_tokens = 20

        response = MagicMock()
        response.content = [text_block, tool_block]
        response.usage = usage
        response.stop_reason = "end_turn"
        response.model = "MiniMax-M2.5"

        mock_instance.messages.create.return_value = response

        client = MiniMaxClient(api_key="sk-test")
        result = client.complete("Test")

        assert result.text == "Response text"

    def test_is_final_true_on_end_turn(self, mock_anthropic):
        _, mock_instance = mock_anthropic
        mock_instance.messages.create.return_value = _mock_anthropic_response(
            stop_reason="end_turn"
        )

        client = MiniMaxClient(api_key="sk-test")
        result = client.complete("Test")
        assert result.is_final is True

    def test_is_final_false_on_max_tokens(self, mock_anthropic):
        """If model hits max_tokens, is_final should be False."""
        _, mock_instance = mock_anthropic
        mock_instance.messages.create.return_value = _mock_anthropic_response(
            stop_reason="max_tokens"
        )

        client = MiniMaxClient(api_key="sk-test")
        result = client.complete("Test")
        assert result.is_final is False

    def test_api_error_propagates(self, mock_anthropic):
        """API errors should propagate (runner handles retry)."""
        _, mock_instance = mock_anthropic
        mock_instance.messages.create.side_effect = anthropic.APIError(
            message="Rate limited",
            request=MagicMock(),
            body=None,
        )

        client = MiniMaxClient(api_key="sk-test")
        with pytest.raises(anthropic.APIError):
            client.complete("Test")

    def test_token_counting(self, mock_anthropic):
        _, mock_instance = mock_anthropic
        mock_instance.messages.create.return_value = _mock_anthropic_response(
            input_tokens=1500,
            output_tokens=2500,
        )

        client = MiniMaxClient(api_key="sk-test")
        result = client.complete("Test")
        assert result.tokens_used == 4000

    def test_long_prompt_passed_through(self, mock_anthropic):
        """Large prompts (task + skills + rubric) should be passed as-is."""
        _, mock_instance = mock_anthropic

        long_prompt = "## Task\n" + "x " * 5000 + "\n## Skills\n" + "y " * 3000

        client = MiniMaxClient(api_key="sk-test")
        client.complete(long_prompt)

        call_args = mock_instance.messages.create.call_args
        assert call_args.kwargs["messages"][0]["content"] == long_prompt


import anthropic  # noqa: E402
