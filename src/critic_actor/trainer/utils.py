"""
Helper functions for Tinker trainers
"""

from typing import Any

from rich.console import Console
from rich.table import Table

console = Console()


def is_better(x: float, y: float, metric: str) -> bool:
    """
    Determine if x is better than y for a given metric
    """
    return x <= y if metric in ["loss"] else x >= y


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


def hide_observations(
    messages: list[dict[str, str]],
    hidden_obs_content: str = "...",
    first_obs_to_show: int = 2,  # e.g., to keep prompt
    last_obs_to_show: int = 1,  # e.g., to keep last observation
) -> list[dict[str, str]]:
    """
    Maybe hide past observations from messages
    """
    user_indices = [
        idx for idx, message in enumerate(messages) if message["role"] in ["user", "tool"]
    ]
    last_message_idx = user_indices[-last_obs_to_show] if last_obs_to_show > 0 else len(messages)
    return [
        {"role": message["role"], "content": hidden_obs_content}
        if (
            message["role"] in ["user", "tool"]
            and (idx >= first_obs_to_show and idx < last_message_idx)
        )
        else message
        for idx, message in enumerate(messages)
    ]
