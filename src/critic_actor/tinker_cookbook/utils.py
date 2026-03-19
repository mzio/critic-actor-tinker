"""
Helper functions for Tinker training
"""

import asyncio
import logging
import time
from collections.abc import Coroutine, Iterable, Sequence
from contextlib import contextmanager
from typing import Any, TypeVar

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

T = TypeVar("T")


# Copied from https://github.com/thinking-machines-lab/tinker-cookbook/blob/22483a6b04400f79da13557a8229bc98b309b026/tinker_cookbook/utils/misc_utils.py#L18
@contextmanager
def timed(key: str, metrics: dict[str, Any]):
    """
    Update metrics with time taken for a given key
    """
    logger.info("Starting %s", key)
    tstart = time.time()
    yield
    logger.info("%s took %.2f seconds", key, time.time() - tstart)
    metrics[f"time/{key}"] = time.time() - tstart


# Copied from https://github.com/thinking-machines-lab/tinker-cookbook/blob/22483a6b04400f79da13557a8229bc98b309b026/tinker_cookbook/utils/misc_utils.py#L58
def split_list(lst: Sequence[T], num_splits: int) -> list[list[T]]:
    """
    Split a sequence into a list of lists, where the sizes are as equal as possible,
    and the long and short lists are as uniformly distributed as possible.

    Args:
        lst: The sequence to split
        num_splits: Number of sublists to create

    Returns:
        A list of sublists with sizes differing by at most 1

    Raises:
        ValueError: If num_splits > len(lst) or num_splits <= 0

    Examples:
        >>> split_list([1, 2, 3, 4, 5], 2)
        [[1, 2, 3], [4, 5]]
        >>> split_list([1, 2, 3, 4, 5], 3)
        [[1, 2], [3, 4], [5]]
    """
    if num_splits <= 0:
        raise ValueError(f"num_splits must be positive, got {num_splits}")
    if num_splits > len(lst):
        raise ValueError(f"Cannot split list of length {len(lst)} into {num_splits} parts")

    edges = np.linspace(0, len(lst), num_splits + 1).astype(int)
    return [list(lst[edges[i] : edges[i + 1]]) for i in range(num_splits)]


# Modified from https://github.com/thinking-machines-lab/tinker-cookbook/blob/22483a6b04400f79da13557a8229bc98b309b026/tinker_cookbook/rl/train.py#L53
async def gather_with_progress(
    coroutines: Iterable[Coroutine[Any, Any, T]],
    per_task_timeout: float | None = 300,  # 10 min default per task
    **pbar_kwargs: Any,
) -> list[T]:
    """
    Run coroutines concurrently with a progress bar that updates as each completes.

    This preserves the order of results (like asyncio.gather) while providing
    real-time progress feedback as individual coroutines complete.

    If per_task_timeout is set, individual tasks that exceed it are cancelled
    and return None.
    """
    coroutine_list = list(coroutines)
    pbar = tqdm(total=len(coroutine_list), **pbar_kwargs)

    async def track(idx: int, coro: Coroutine[Any, Any, T]) -> T | None:
        try:
            if per_task_timeout is not None:
                result = await asyncio.wait_for(coro, timeout=per_task_timeout)
            else:
                result = await coro
            pbar.update(1)
            return result
        except asyncio.TimeoutError:
            logger.warning("Task %d timed out after %.0fs, skipping", idx, per_task_timeout)
            pbar.update(1)
            return None

    try:
        results = await asyncio.gather(*[track(i, coro) for i, coro in enumerate(coroutine_list)])
    finally:
        pbar.close()

    return results

