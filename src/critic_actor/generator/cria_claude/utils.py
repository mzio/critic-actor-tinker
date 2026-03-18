"""
Helper functions for Critic-Actor on top of the Claude Agent SDK
"""

import json
from typing import Any

from rich import print as rich_print

from claude_agent_sdk import (
    ClaudeSDKClient,
    AssistantMessage,
    ResultMessage,
    ThinkingBlock,
    TextBlock,
    ToolUseBlock,
)

from critic_actor.llm_handlers.types import ActionFromLLM
from .types import ClaudeAgentResponse


def convert_tools_dict_to_str(
    tools: list[dict[str, Any]],
    include_type: bool = False,  # 'type': 'function'
) -> str:
    """
    Convert list of tool dicts to a markdown string
    """
    
    def get_markdown_from_dict(
        tool_dict: dict[str, Any],
        include_type: bool = False,
    ) -> str:
        """
        Convert a single tool dict to a markdown string
        """
        parts: list[str] = [f'name: "{tool_dict['name']}"']
        keys = [k for k in tool_dict.keys() if k != "type" and k != "name"]
        if include_type:
            keys += ["type"]
        for k in keys:
            v = json.dumps(tool_dict[k]) if isinstance(tool_dict[k], (dict, list)) else tool_dict[k]
            parts.append(f"- {k}: {v}")

        return "\n".join(parts)

    return "\n\n".join([get_markdown_from_dict(t, include_type) for t in tools])


def get_prompt_from_messages(
    messages: list[dict[str, Any]],
    system_prompt: str | None = None,
    delimiter: str = "\n\n",
) -> str:
    """
    Convert a list of chat messages into a single prompt string for client.query().

    For single-turn use, we format the conversation history as a prompt.
    """
    parts: list[str] = []
    if system_prompt:
        parts.append(f"[System]\n{system_prompt}")

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "tool" or msg.get("type") == "function_call_output":
            # call_id = msg.get("call_id", "unknown")
            call_id = msg.get("call_id", None)
            call_id = f" ({call_id})" if call_id else ""
            parts.append(f"[Tool Result{call_id}]\n{content}")

        elif role == "assistant":
            parts.append(f"[Assistant]\n{content}")

        elif role == "user":
            parts.append(f"[User]\n{content}")

        elif role == "system":
            parts.append(f"[System]\n{content}")

    return delimiter.join(parts)


async def sample_client_response(
    client: ClaudeSDKClient,
    prompt: str,
    **kwargs: Any,
) -> list[ClaudeAgentResponse]:
    """
    Sample a response from the Claude Agent SDK client
    """
    response = ClaudeAgentResponse()
    await client.query(prompt)

    async for msg in client.receive_response():
        if isinstance(msg, AssistantMessage):
            response.assistant_messages.append(msg)
        elif isinstance(msg, ResultMessage):
            response.result = msg
            if msg.usage:
                response.usage = {
                    "input_tokens": msg.usage.get("input_tokens", 0),
                    "output_tokens": msg.usage.get("output_tokens", 0),
                }
            if getattr(msg, "total_cost_usd", None):
                response.cost = msg.total_cost_usd
    return response


def get_actions_from_response(
    response: ClaudeAgentResponse,
    **kwargs: Any,
) -> list[ActionFromLLM]:
    """
    Get actions from a Claude Agent SDK response
    """
    actions: list[ActionFromLLM] = []
    for _, block in enumerate(response.output):
        # ActionFromLLM tool call default kwargs
        tool_call_kwargs: dict[str, Any] = {"call_id": None, "name": None, "arguments": None}

        if isinstance(block, ThinkingBlock):
            actions.append(
                ActionFromLLM(
                    role="assistant",
                    type="reasoning",
                    text=block.thinking,
                    **tool_call_kwargs,
                )
            )
        elif isinstance(block, TextBlock):
            actions.append(
                ActionFromLLM(
                    role="assistant",
                    type="message",
                    text=block.text,
                    **tool_call_kwargs,
                )
            )
        elif isinstance(block, ToolUseBlock) and block.name == "StructuredOutput":
            actions.append(
                ActionFromLLM(
                    role="assistant",
                    type="function_call",
                    text=json.dumps(block.input),
                    call_id=block.id,
                    name=block.name,
                    arguments=block.input,
                )
            )
    return actions


def display_actions(actions: list[ActionFromLLM], base_color: str) -> None:
    """
    Display actions in a readable format
    """
    for action_idx, action in enumerate(actions):
        if action.type == "reasoning":
            style = f"italic {base_color}"
        elif action.type == "message":
            style = f"{base_color}"
        elif action.type == "function_call":
            style = f"bold {base_color}"
        rich_print(
            f"[{style}] {action_idx}. ({action.type})\n{action.text} [/{style}]"
        )