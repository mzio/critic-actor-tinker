"""
Utility functions for the GDPval environment.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from critic_actor.llm_handlers.types import ActionFromLLM

logger = logging.getLogger(__name__)


def format_exec_result(
    stdout: str | None,
    stderr: str | None,
    return_code: int,
    max_chars: int = 16000,
) -> str:
    """Format stdout/stderr/exit_code into a string for the model.

    Truncates output to ``max_chars`` to prevent context window overflow.
    """
    parts = []
    if stdout:
        text = stdout[:max_chars] if len(stdout) > max_chars else stdout
        if len(stdout) > max_chars:
            text += f"\n... (truncated {len(stdout) - max_chars} chars)"
        parts.append(text)
    if stderr:
        text = stderr[:max_chars] if len(stderr) > max_chars else stderr
        if len(stderr) > max_chars:
            text += f"\n... (truncated {len(stderr) - max_chars} chars)"
        parts.append(f"[stderr]\n{text}")
    if return_code != 0:
        parts.append(f"[exit code: {return_code}]")

    return "\n".join(parts) if parts else "(no output)"


def extract_action(parsed_actions: list[ActionFromLLM]) -> tuple[str | None, str | None]:
    """Extract an execute_python or submit action from parsed actions.

    Returns:
        ``(code_or_none, action_type)`` where ``action_type`` is
        ``"execute_python"``, ``"submit"``, or ``None``.
    """
    for action in parsed_actions:
        if action.type == "function_call":
            if action.name == "execute_python":
                code = (action.arguments or {}).get("code", "")
                return code, "execute_python"
            elif action.name == "submit":
                return None, "submit"

    # Fall back to text message (treat as code)
    for action in parsed_actions:
        if action.type == "message" and action.text:
            return action.text, "execute_python"

    return None, None


def collect_deliverables(
    workdir: Path,
    reference_filenames: set[str],
) -> list[Path]:
    """Collect files created by the model (excluding reference files).

    Args:
        workdir: The working directory for this episode.
        reference_filenames: Set of filenames that were provided as reference files.

    Returns:
        List of paths to deliverable files created by the model.
    """
    deliverables = []
    for path in workdir.rglob("*"):
        if not path.is_file():
            continue
        # Skip reference files
        if path.parent == workdir and path.name in reference_filenames:
            continue
        # Skip hidden files and __pycache__ (use relative parts to avoid
        # false positives from dot-prefixed dirs in the workdir path itself)
        rel_parts = path.relative_to(workdir).parts
        if any(part.startswith(".") or part == "__pycache__" for part in rel_parts):
            continue
        deliverables.append(path)
    return sorted(deliverables)
