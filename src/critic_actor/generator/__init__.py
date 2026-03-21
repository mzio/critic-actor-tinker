"""
Generators for LLM-based rollouts
"""

from collections.abc import Callable
from functools import partial
from typing import Any

from .base import TinkerGenerator


def get_generator_constructor(
    name: str,
    **kwargs: Any,
) -> Callable[..., Any]:
    """
    Get a (partially initialized) TinkerGenerator constructor by name

    e.g., if **kwargs has all the necessary arguments (it won't in most cases),
    we can get the TinkerGenerator object via:

    ```python
    generator_ctor = get_generator_constructor(**generator_cfg)
    generator = generator_ctor()
    ```
    """

    if name in ["default"]:
        from .base import TinkerGenerator
        return partial(TinkerGenerator, **kwargs)

    if name in ["cria_claude", "critic_actor_claude"]:
        from .cria_claude import CriticActorClaudeGenerator
        return partial(CriticActorClaudeGenerator, **kwargs)

    if name in ["chat_claude"]:
        from .chat_claude import ChatClaudeGenerator
        return partial(ChatClaudeGenerator, **kwargs)

    else:
        raise NotImplementedError(f"Sorry, generator {name} is not implemented yet.")


__all__ = [
    "get_generator_constructor",
    "TinkerGenerator",
]
