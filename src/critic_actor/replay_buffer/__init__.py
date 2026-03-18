"""
Replay buffer for storing episode steps (at minimum, (state, action, advantage) tuples)
"""

from typing import Any

from transformers import PreTrainedTokenizerBase

from .base import ReplayBuffer
from .types import EpisodeStep, Trajectory, TrajectoryGroup, MeanCenteredTrajectoryGroup


def get_replay_buffer(
    name: str,
    hf_tokenizer: PreTrainedTokenizerBase,
    **kwargs: Any,
) -> ReplayBuffer:
    """
    Get a replay buffer by name
    """
    if name == "default":
        return ReplayBuffer(**kwargs)

    else:
        raise NotImplementedError(f"Sorry, replay buffer '{name}' is not implemented yet.")


__all__ = [
    "ReplayBuffer",
    "EpisodeStep",
    "Trajectory",
    "TrajectoryGroup",
    "MeanCenteredTrajectoryGroup",
]
