"""
Types for Critic-Actor on top of the Claude Agent SDK
"""

from dataclasses import dataclass, field
from typing import Any
from pydantic import BaseModel

from claude_agent_sdk import AssistantMessage, ResultMessage


class ToolCallMessage(BaseModel):
    """
    Tool call message
    """
    reasoning: str | None
    tool_call: str


class ClaudeResponseForCriticActor(BaseModel):
    """
    Structured output for parsing potential generations
    from ClaudeAgentSDK Client
    """
    messages: list[ToolCallMessage]


@dataclass
class ClaudeAgentResponse:
    """
    Response container that mirrors the structure expected by get_actions().
    Collects AssistantMessages from the Claude Agent SDK stream.
    """
    assistant_messages: list[AssistantMessage] = field(default_factory=list)
    result: ResultMessage | None = None
    usage: dict[str, int] | None = None
    cost: float | None = 0.0

    @property
    def output(self) -> list[Any]:
        """Flatten all content blocks from all assistant messages."""
        blocks = []
        for msg in self.assistant_messages:
            blocks.extend(msg.content)
        return blocks
