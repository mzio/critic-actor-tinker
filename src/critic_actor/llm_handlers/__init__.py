"""
LLM classes and types
"""

from typing import Any

from .base import LLM
from .openai import (
    OpenAIResponsesLLM,
    AsyncOpenAIResponsesLLM,
    Response,
)
from .claude_agent_sdk import (
    ClaudeQueryLLM,
    ClaudeClientLLM,
    ClaudeAgentResponse,
)
from .tinker import TinkerCompleter
from .types import ActionFromLLM


def load_llm(
    name: str,
    model_config: dict[str, Any],
    is_async: bool = False,
    **kwargs: Any,
) -> LLM:
    """
    Load LLM
    """

    if name == "tinker":
        return TinkerCompleter(**model_config)
    
    if name == "openai":
        if is_async:
            return AsyncOpenAIResponsesLLM(**model_config)
        else:
            return OpenAIResponsesLLM(**model_config)

    if name == "claude_query":
        return ClaudeQueryLLM(**model_config)

    if name == "claude_client":
        return ClaudeClientLLM(**model_config)

    raise ValueError(f"Invalid model name: {name}")


__all__ = [
    "load_llm",
    "LLM", "ActionFromLLM",
    "OpenAIResponsesLLM", "AsyncOpenAIResponsesLLM", "Response",
    "ClaudeQueryLLM", "ClaudeClientLLM", "ClaudeAgentResponse",
    "TinkerCompleter",
]
