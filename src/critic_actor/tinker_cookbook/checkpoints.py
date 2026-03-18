"""
Functions for saving checkpoints and getting sampling clients
"""

import asyncio
import json
import logging
import os
from typing import Any, Literal

import tinker
from tinker_cookbook.utils.trace import update_scope_context

from .utils import timed

logger = logging.getLogger(__name__)


# Modified from https://github.com/thinking-machines-lab/tinker-cookbook/blob/f1daee9d1ce0a8102e0e0ed8151f99d98919a503/tinker_cookbook/checkpoint_utils.py#L233
async def save_checkpoint_async(
    training_client: tinker.TrainingClient,
    name: str,
    log_path: str,
    loop_state: dict[str, Any],
    kind: Literal["state", "sampler", "both"] = "state",
) -> dict[str, str]:
    """Save model checkpoint.
    Args:
        training_client: Training client to save from
        name: Name for the checkpoint
        log_path: Path to the log directory, where we can find checkpoints.jsonl file
    Returns:
        Path to the saved checkpoint
    """
    futures = {}
    if kind in ["state", "both"]:
        futures["state"] = await training_client.save_state_async(name)
    if kind in ["sampler", "both"]:
        futures["sampler"] = await training_client.save_weights_for_sampler_async(name)

    # results = {k: await v.result_async() for k, v in futures.items()}
    results = await asyncio.gather(*(v.result_async() for v in futures.values()))
    results = dict(zip(futures.keys(), results))

    paths = {k + "_path": v.path for k, v in results.items()}
    update_scope_context(paths)
    logger.info("Saved checkpoints: %s", paths)
    full_dict = {"name": name, **loop_state, **paths}
    with open(os.path.join(log_path, "checkpoints.jsonl"), "a") as f:
        f.write(json.dumps(full_dict) + "\n")

    return paths


# Modified from https://github.com/thinking-machines-lab/tinker-cookbook/blob/22483a6b04400f79da13557a8229bc98b309b026/tinker_cookbook/rl/train.py#L714
async def save_checkpoint_and_get_sampling_client(
    training_client: tinker.TrainingClient,
    i_batch: int,
    log_path: str,
    save_every: int,
    start_batch: int = 0,
    checkpoint_name: str | None = None,
) -> tuple[tinker.SamplingClient, dict[str, Any]]:
    """
    Save checkpoint and get sampling client
    """
    metrics = {}
    name = f"{checkpoint_name}_{i_batch:06d}" if checkpoint_name else f"{i_batch:06d}"
    with timed("save_checkpoint", metrics):
        if save_every > 0 and i_batch > start_batch and i_batch % save_every == 0:
            path_dict = await save_checkpoint_async(
                training_client=training_client,
                name=name,
                log_path=log_path,
                loop_state={"batch": i_batch},
                kind="both",
            )
            return training_client.create_sampling_client(path_dict["sampler_path"]), metrics
        return await training_client.save_weights_and_get_sampling_client_async(), metrics
