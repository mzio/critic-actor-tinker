"""
Helper functions for Tinker-based generators
"""

import logging
from typing import Any

from critic_actor.environments import EnvironmentState
from critic_actor.replay_buffer import ReplayBuffer

DEBUG_COLS = ["batch_id", "split", "try_step", "generation_id", "unique_data_sample_id"]

logger = logging.getLogger(__name__)


def get_response_content(msg: dict[str, Any]) -> str:
    """
    Get message content from an Environment response message
    """
    return msg["output"] if msg.get("output") else msg["content"]


def remove_prefix_messages(
    messages: list[dict[str, Any]],
    default_context: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Remove system prompt and default context from messages
    """
    # First remove system prompt
    msgs = [msg for msg in messages if msg.get("role", "") != "system"]

    # Then deal with default context
    # -> Remove a contiguous list of messages whenever it appears
    _default_context = default_context or []
    if len(_default_context) == 0:
        return msgs
    # Otherwise, remove default context messages kinda like Lomuto's
    l_idx = 0
    r_idx = 0
    w_len = len(_default_context)
    # for r_idx in range(len(msgs) - w_len + 1):
    while r_idx < len(msgs):
        if msgs[r_idx : r_idx + w_len] == _default_context:
            # Swap the default context entries
            swapped_msgs = msgs[r_idx : r_idx + w_len], msgs[l_idx : l_idx + w_len]
            msgs[l_idx : l_idx + w_len], msgs[r_idx : r_idx + w_len] = swapped_msgs
            l_idx += w_len
            r_idx += w_len
        else:
            r_idx += 1
    return msgs[l_idx:]


def get_past_try_messages(
    state: EnvironmentState,
    split: str | None = None,
    timestep: int | None = None,
    try_step: int | None = None,
    replay_buffer: ReplayBuffer | None = None,
    default_context: list[dict[str, Any]] | None = None,
    **get_past_episode_steps_kwargs: Any,
) -> list[dict[str, Any]]:
    """
    Get past try messages to prepend to context (default base behavior)
    """
    past_rollout_messages: list[dict[str, str]] = []
    default_context = default_context or []

    # Sanity-check timestep and try_step
    if timestep is not None:
        assert timestep == state.timestep, (
            f"timestep={timestep} != state.timestep={state.timestep}"
        )
    if try_step is not None:
        assert try_step == state.try_step, (
            f"try_step={try_step} != state.try_step={state.try_step}"
        )
    timestep = timestep or state.timestep
    try_step = try_step or state.try_step
    # Get split from arguments or state
    split = split or getattr(state, "split", None)
    assert split == getattr(state, "split", None), "split != state.split"
    assert split is not None, "split must be provided"

    # Default (base) behavior: append past rollout steps
    if timestep == 0 and try_step > 0 and replay_buffer is not None:
        # Get past rollout steps and messages from replay buffer
        # -> As we append prior rollouts to each subsequent try, we only need to get the last
        #    try's rollout to add *all* prior rollouts to context
        past_rollout_steps = replay_buffer.get_past_episode_steps(
            split=split,
            batch_id=state.batch_id,  # specifically for standard multi-try, should match on current batch_id
            try_step=try_step - 1,
            generation_id=state.generation_id,
            unique_data_sample_id=state.sample_id,
            **get_past_episode_steps_kwargs,
        )
        try:
            past_rollout_messages = replay_buffer.get_messages_from_steps(
                past_rollout_steps,
                timestep=-1,  # by default, get the last-step messages,
                full_rollout=True,  # i.e., the full trajectory
            )
        except Exception as e:
            logger.error(
                "Error on:\nreplay_buffer.get_messages_from_steps(past_rollout_steps)"
            )
            logger.error("%s: %s", e.__class__.__name__, e)
            logger.error("past_rollout_steps: %s", past_rollout_steps)
            _df = replay_buffer.pd_df_buffer
            _df_debug = _df[
                (_df["generation_id"] == state.generation_id)
                & (_df["unique_data_sample_id"] == state.sample_id)
            ][DEBUG_COLS]
            logger.error("debug_cols: %s", DEBUG_COLS)
            logger.error("_df_debug: %s", _df_debug)
            breakpoint()
            raise e
        # Hacky check, but remove system prompt and default context if in past_rollout_messages
        past_rollout_messages = remove_prefix_messages(past_rollout_messages, default_context)
    return past_rollout_messages


if __name__ == "__main__":
    # Do some simple print-based tests
    test_messages = [
        # {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
        {"role": "user", "content": "What is the capital of Pennsylvania?"},
        {"role": "assistant", "content": "The capital of Pennsylvania is Harrisburg."},
    ]
    test_default_context = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    for _idx, _msg in enumerate(test_messages):
        print(f"test_messages[{_idx}]: {_msg}")
    print("---" * 10)
    test_new_messages = remove_prefix_messages(test_messages, test_default_context)
    for _idx, _msg in enumerate(test_new_messages):
        print(f"test_new_messages[{_idx}]: {_msg}")
    print("---" * 10)
