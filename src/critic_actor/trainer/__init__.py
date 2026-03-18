"""
Tinker trainers for SFT and RL
"""

from typing import Any

from .base import TinkerTrainer


def get_trainer(
    name: str,
    **kwargs: Any,
) -> TinkerTrainer:
    """
    Get a Tinker trainer by name
    """
    if name in ["rl"]:
        from .rl import RlTinkerTrainer

        return RlTinkerTrainer(**kwargs)

    else:
        raise NotImplementedError(f"Sorry, trainer {name} not implemented yet.")
