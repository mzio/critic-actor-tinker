"""
Helper functions for displaying data
"""

import logging
from functools import partial
from typing import Any, TypeVar

from omegaconf import DictConfig
from rich import box
from rich.console import Console
from rich.errors import MarkupError
from rich.panel import Panel
from rich.table import Table

from transformers import PreTrainedTokenizerBase
from transformers.generation.streamers import TextStreamer

logger = logging.getLogger(__name__)
console = Console()


T = TypeVar("T")


class RichTextStreamer(TextStreamer):
    """
    A streamer that prints the text in a rich format
    """

    def on_finalized_text(self, text: str, stream_end: bool = False) -> None:
        rich_print_messages(msg_text=text, flush=True, end="" if not stream_end else None)


def display_metrics(
    metrics: dict[str, Any],
    title: str | None = None,
    style: str = "bright_yellow",
) -> None:
    """
    Display metrics in a table
    """
    table = Table(title=title, style=style)
    table.add_column("Metric", justify="left", style=style)
    table.add_column("Value", justify="left", style=f"bold {style}")
    for k, v in metrics.items():
        table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
    console.print(table)


def rich_print_messages(
    msg_text: str,
    bos_token: str = "<|im_start|>",
    eos_token: str = "<|im_end|>\n",
    tool_call_bos_token: str = "<tool_call>",
    tool_call_eos_token: str = "</tool_call>",
    tool_response_bos_token: str = "<tool_response>",
    tool_response_eos_token: str = "</tool_response>",
    # Silly coloring
    system_color: str = "bright_yellow",
    user_color: str = "bright_red",
    assistant_color: str = "bright_cyan",
    tool_call_color: str = "dodger_blue1",
    tool_response_color: str = "bright_magenta",
    display_messages: bool = True,
    **rich_print_kwargs: Any,
) -> str:
    """
    Print chat-templated messages in silly colors
    """
    # Split into messages
    msgs = msg_text.split(eos_token)

    system_bos = f"{bos_token}system"
    user_bos = f"{bos_token}user"
    assistant_bos = f"{bos_token}assistant"

    for ix, msg in enumerate(msgs):
        # system prompt
        if msg.startswith(system_bos):
            msgs[ix] = f"[{system_color}]{msg}[/{system_color}]"
        # user messages
        elif msg.startswith(user_bos):
            msgs[ix] = f"[{user_color}]{msg}[/{user_color}]"
        # assistant messages
        elif msg.startswith(assistant_bos):
            msgs[ix] = f"[{assistant_color}]{msg}[/{assistant_color}]"

        # tool calls
        if tool_call_bos_token in msgs[ix] and tool_call_eos_token in msgs[ix]:
            msgs[ix] = msgs[ix].replace(
                tool_call_bos_token, f"[{tool_call_color}]{tool_call_bos_token}"
            )
            msgs[ix] = msgs[ix].replace(
                tool_call_eos_token, f"{tool_call_eos_token}[/{tool_call_color}]"
            )
        # tool responses
        if tool_response_bos_token in msgs[ix] and tool_response_eos_token in msgs[ix]:
            msgs[ix] = msgs[ix].replace(
                tool_response_bos_token, f"[{tool_response_color}]{tool_response_bos_token}"
            )
            msgs[ix] = msgs[ix].replace(
                tool_response_eos_token, f"{tool_response_eos_token}[/{tool_response_color}]"
            )

        # "Final Answer" replies
        if "# RESULT:" in msgs[ix]:
            # MZ 02/13/2026 NOTE (to self): red-green colorblind...
            _color = "bright_yellow" if "INCORRECT" in msgs[ix] else "bright_cyan"
            msgs[ix] = f"[bold {_color}]{msg}[/bold {_color}]"

    msgs_text = eos_token.join(msgs)
    if display_messages:
        try:
            console.print(msgs_text, **rich_print_kwargs)
        except MarkupError as e:
            logger.error(f"rich.errors.MarkupError: {e}")
            print(msgs_text)
            logger.error(f"{e.__class__.__name__}: {e}")
    return msgs_text


def display_state_action_next_obs(
    *,
    generation_id: int,
    state_messages: list[dict[str, Any]],
    action_messages: list[dict[str, Any]],
    next_obs_messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    hf_tokenizer: PreTrainedTokenizerBase,
    cfg: DictConfig | None = None,
    generator: Any | None = None,
    header_text: str | None = None,
    panel_content: str | None = None,
    group_rewards: list[float] | None = None,
    state: Any | None = None,  # Environment state object for dynamic attribute extraction
    **rich_print_kwargs: Any,
) -> None:
    """
    Display the state, action, and next observations in a rich format
    """
    # Silly coloring to differentiate between generations
    _base_color = f"color({(generation_id + 1) % 8 + 8})"  # +8 for bright colors
    _bold_color = f"bold {_base_color}"

    path_cls = generator or cfg  # for run url's and commands and other stuff

    # Convenience variables and constructors
    header_text = header_text or f"Generation {generation_id}"
    _apply_chat_template = partial(hf_tokenizer.apply_chat_template, tokenize=False)
    _get_panel = partial(Panel, border_style=_bold_color)

    # Display current state
    rich_state_text = rich_print_messages(
        msg_text=_apply_chat_template(state_messages, add_generation_prompt=True, tools=tools),
        user_color=_base_color,
        assistant_color=_bold_color,
        display_messages=False,
        tool_response_color=f"italic {_base_color}",
        **rich_print_kwargs,
    )
    # Display action
    rich_action_text = rich_print_messages(
        msg_text=_apply_chat_template(action_messages, add_generation_prompt=False),
        user_color=f"italic {_base_color}",
        assistant_color=f"italic {_bold_color}",
        tool_call_color=f"italic {_bold_color}",
        display_messages=False,
        **rich_print_kwargs,
    )
    # Display next observation
    rich_next_obs_text = rich_print_messages(
        msg_text=_apply_chat_template(next_obs_messages, add_generation_prompt=False),
        user_color=f"dim italic {_base_color}",
        assistant_color=f"dim italic {_bold_color}",
        tool_response_color=f"dim italic {_base_color}",
        display_messages=False,
        **rich_print_kwargs,
    )

    # Display states, actions, and next observations
    try:
        console.print(_get_panel(rich_state_text, title=f"{header_text} [State]"))
    except Exception as e:
        logger.error(f"{e.__class__.__name__}: {e}")
        print(f"{header_text} [State]\n{rich_state_text}")
    try:
        console.print(_get_panel(rich_action_text, title=f"{header_text} [Action]"))
    except Exception as e:
        logger.error(f"{e.__class__.__name__}: {e}")
        print(f"{header_text} [Action]\n{rich_action_text}")
    try:
        console.print(_get_panel(rich_next_obs_text, title=f"{header_text} [Next Obs]"))
    except Exception as e:
        logger.error(f"{e.__class__.__name__}: {e}")
        print(f"{header_text} [Next Obs]\n{rich_next_obs_text}")

    # Display panel content
    if panel_content is None:  # Default to group_rewards and run info
        panel_content_list = (
            [f"Rewards: [bright_green][{str(group_rewards)}][/bright_green]"]
            if group_rewards is not None
            else []
        )
        # Add environment-specific state information dynamically
        if state is not None:
            state_info_list = []
            # Common useful attributes across environments (non-exhaustive)
            display_attrs = {
                "answer": ("Answer", "bright_yellow"),
                "query": ("Query", "cyan"),
                "words": ("Words", "dim"),
                "recall_distance": ("Recall Distance", "magenta"),
            }
            for attr, (label, color) in display_attrs.items():
                if hasattr(state, attr):
                    value = getattr(state, attr)
                    if value is not None:
                        # Format the value appropriately
                        value_str = str(value)
                        state_info_list.append(f"{label}: [{color}]{value_str}[/{color}]")

            # Add any additional metadata that might be useful
            if hasattr(state, "metadata") and state.metadata:
                for key, value in state.metadata.items():
                    if key not in [
                        "correct",
                        "total",
                        "reward",
                        "done",
                        "truncated",
                        "updated_try_step",
                    ]:
                        state_info_list.append(f"{key}: [dim]{value}[/dim]")

            if state_info_list:
                panel_content_list.extend(state_info_list)

        panel_content_list.extend(
            [
                f"Run url: [link={path_cls.run_url}]{path_cls.run_url}[/link]",
                f"Run cmd: [bright_blue]{path_cls.run_cmd}[/bright_blue]",
            ]
        )
        if path_cls.last_generated_data_url is not None:
            panel_content_list.append(
                f"Generated data url: "
                f"[link={path_cls.last_generated_data_url}]{path_cls.last_generated_data_url}[/link]"
            )
        if path_cls.last_replay_buffer_path is not None:
            panel_content_list.append(
                f"Replay buffer path: "
                f"[bright_yellow]{path_cls.last_replay_buffer_path}[/bright_yellow]"
            )
        panel_content = f"[bold]{'\n'.join(panel_content_list)}[/bold]"
    try:
        console.print(Panel(panel_content, title=header_text, box=box.HORIZONTALS))
    except Exception as e:
        logger.error(f"{e.__class__.__name__}: {e}")
        print(f"{header_text}\n{panel_content}")
