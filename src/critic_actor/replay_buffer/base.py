"""
Default replay buffer for storing training samples
"""

import asyncio
import logging
from copy import deepcopy
from typing import Any

import pandas as pd
from datasets import Dataset

from .types import EpisodeStep, Trajectory, TrajectoryGroup

logger = logging.getLogger(__name__)


DEFAULT_DUPLICATE_KEYS = [
    "state_len",
    "done",
    "truncated",
    "split",
    "batch_id",  # "timestep", "try_step",
    "unique_data_sample_id",
    "generation_id",
    "return_is_computed",
    "advantage_is_computed",
    "is_train",
]


class ReplayBuffer:
    """
    Replay buffer for storing state, action, next_obs, return, advantage
    """

    def __init__(
        self,
        max_size: int = 1e10,
        remove_duplicates: bool | None = None,
        duplicate_keys: list[str] | None = None,
        debug: bool = False,
    ) -> None:
        self.max_size = int(max_size)
        self.remove_duplicates = remove_duplicates
        self.duplicate_keys = duplicate_keys or DEFAULT_DUPLICATE_KEYS

        self.buffer: list[dict[str, Any]] = []
        self.embeddings: list[list[float]] = []  # will convert these to torch.Tensor later
        # The only things we need to filter on are:
        # - split, batch_id, try_step, data_sample_id ?
        self.hf_ds_buffer: Dataset | None = None  # Initialize after saving or loading from disk
        self.pd_df_buffer: pd.DataFrame | None = None  # Optionally use for quick filtering?
        self.iterable_keys = []
        self.debug = debug

    def _df_features_to_list(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert all iterable features in df to lists.
        Rows where the value is None/NaN are left as-is.
        """
        for k in self.iterable_keys:
            if k not in df.columns:
                continue
            df[k] = df[k].apply(lambda x: x.tolist() if hasattr(x, "tolist") else x)
        return df

    def add_episode_step(self, episode_step: EpisodeStep) -> None:
        """
        Add an episode step to the replay buffer
        """
        dict_step = {k: v for k, v in episode_step.model_dump().items() if v is not None}
        self.buffer.append(dict_step)
        self.iterable_keys.extend(
            k for k in dict_step if isinstance(dict_step[k], list) and k not in self.iterable_keys
        )

    def add_trajectory(self, trajectory: Trajectory) -> None:
        """
        Add all episode steps in a trajectory to the replay buffer
        """
        for episode_step in trajectory.episode_steps:
            self.add_episode_step(episode_step)

    def add_trajectory_group(self, trajectory_group: TrajectoryGroup) -> None:
        """
        Add all episode steps, in all trajectories in a trajectory group, to the replay buffer
        """
        for trajectory in trajectory_group.trajectories:
            self.add_trajectory(trajectory)

    def save_to_hf_dataset(
        self,
        save_path: str,
        remove_duplicates: bool | None = None,
        duplicate_keys: list[str] | None = None,
    ) -> None:
        """
        Save all samples in self.buffer to a Hugging Face dataset
        """
        assert len(self.buffer) > 0, "No samples to save"
        remove_duplicates = remove_duplicates or self.remove_duplicates

        self.hf_ds_buffer = Dataset.from_list(self.buffer)
        # Remove duplicates from dataset
        if remove_duplicates:
            self.pd_df_buffer = self.get_df_from_dataset(
                self.hf_ds_buffer,
                remove_duplicates,
                duplicate_keys,
            )
            self.hf_ds_buffer = Dataset.from_pandas(self.pd_df_buffer)
        else:
            self.pd_df_buffer = self.hf_ds_buffer.to_pandas()
        self.hf_ds_buffer.save_to_disk(save_path)
        # Update iterable keys
        self.iterable_keys.extend(
            k
            for k in self.hf_ds_buffer.features
            if isinstance(self.hf_ds_buffer[0][k], list) and k not in self.iterable_keys
        )

    async def save_to_hf_dataset_async(
        self,
        save_path: str,
        remove_duplicates: bool | None = None,
        duplicate_keys: list[str] | None = None,
    ) -> None:
        """
        Save all samples in self.buffer to a Hugging Face dataset asynchronously
        """
        assert len(self.buffer) > 0, "No samples to save"
        data = list(self.buffer)

        def _save():
            self.hf_ds_buffer = Dataset.from_list(data)
            if remove_duplicates:
                self.pd_df_buffer = self.get_df_from_dataset(
                    self.hf_ds_buffer,
                    remove_duplicates,
                    duplicate_keys,
                )
                self.hf_ds_buffer = Dataset.from_pandas(self.pd_df_buffer)
            else:
                self.pd_df_buffer = self.hf_ds_buffer.to_pandas()
            self.hf_ds_buffer.save_to_disk(save_path)

        await asyncio.to_thread(_save)

    def get_df_from_dataset(
        self,
        dataset: Dataset,
        remove_duplicates: bool | None = None,
        duplicate_keys: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Get a DataFrame from a Hugging Face dataset
        -> If remove_duplicates, remove duplicate rows based on everything but `pgic_step_idx`
           by default (see PolicyGradientInContextHFGenerator in generator/pgic.py)
        """
        duplicate_keys = duplicate_keys or self.duplicate_keys
        if remove_duplicates:
            return dataset.to_pandas().drop_duplicates(subset=duplicate_keys).reset_index(drop=True)
        return dataset.to_pandas()

    def load_from_hf_dataset(self, load_path: str, verbose: bool = False) -> None:
        """
        Load all samples from a Hugging Face dataset
        """
        self.hf_ds_buffer = Dataset.load_from_disk(load_path)
        self.pd_df_buffer = self.hf_ds_buffer.to_pandas()
        # Get iterable keys, convert df columns to lists, and initialize buffer
        self.iterable_keys.extend(
            k
            for k in self.hf_ds_buffer.features
            if isinstance(self.hf_ds_buffer[0][k], list) and k not in self.iterable_keys
        )
        self.buffer = self.hf_ds_buffer.to_list()
        # self.buffer = self.pd_df_buffer.to_dict(orient="records")

        if verbose:
            logger.info("Loaded %d samples from %s", len(self.buffer), load_path)
            logger.info("Sample keys: %s", str(self.buffer[0].keys()))

    def get_past_episode_steps(self, **kwargs: Any) -> list[EpisodeStep]:
        """
        Get episode steps from a prior rollout (override in child classes)
        """
        return self._get_rollout_episode_steps(**kwargs)

    def _get_rollout_episode_steps(
        self,
        **filter_kwargs: Any,
    ) -> list[EpisodeStep]:
        """
        Get episode steps for a prior rollout, given unique data sample id
        """
        df_rollout = self.get_df_for_rollout(**filter_kwargs)
        step_dicts = self._df_features_to_list(df_rollout).to_dict(orient="records")
        return [EpisodeStep(**step_dict) for step_dict in step_dicts]

    def get_df_for_rollout(
        self,
        split: str | None = None,
        unique_data_sample_id: int | None = None,
        batch_id: int | None = None,
        try_step: int | None = None,
        timestep: int | None = None,
        generation_id: int | None = None,
        less_than_kwargs: dict[str, Any] | None = None,
        more_than_kwargs: dict[str, Any] | None = None,
        not_equal_kwargs: dict[str, Any] | None = None,
        leq_than_kwargs: dict[str, Any] | None = None,
        geq_than_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Get the prior rollouts as a DataFrame
        """
        # Could use kwargs, but better to make these arguments explicit
        filter_kwargs = {
            "split": split,
            "unique_data_sample_id": unique_data_sample_id,
            "batch_id": batch_id,
            "try_step": try_step,
            "generation_id": generation_id,
            "timestep": timestep,
        }
        filter_kwargs.update(kwargs)
        for k in filter_kwargs:
            assert k in self.pd_df_buffer.columns, f"Invalid filter keyword argument: {k}"
        # Build selection mask for filtering
        mask = True
        for k, v in filter_kwargs.items():
            if v is not None:
                mask = mask & (self.pd_df_buffer[k] == v)
        # Optionally filter on <, >, or != certain values
        if less_than_kwargs is not None:
            for k, v in less_than_kwargs.items():
                mask = mask & (self.pd_df_buffer[k] < v)
        if more_than_kwargs is not None:
            for k, v in more_than_kwargs.items():
                mask = mask & (self.pd_df_buffer[k] > v)
        if not_equal_kwargs is not None:
            for k, v in not_equal_kwargs.items():
                mask = mask & (self.pd_df_buffer[k] != v)
        # Kinda heinous but ditto for <= and >=
        if leq_than_kwargs is not None:
            for k, v in leq_than_kwargs.items():
                mask = mask & (self.pd_df_buffer[k] <= v)
        if geq_than_kwargs is not None:
            for k, v in geq_than_kwargs.items():
                mask = mask & (self.pd_df_buffer[k] >= v)

        return self.pd_df_buffer[mask].copy(deep=True).reset_index(drop=True)

    def get_messages_from_steps(
        self,
        steps: list[EpisodeStep],
        timestep: int = -1,  # by default, get the last-step messages,
        full_rollout: bool = True,  # i.e., the full trajectory
    ) -> list[dict[str, str]]:
        """
        Return the combined state, action, and next_obs chat for the last step of a trajectory.
        This should correspond to the "trajectory-to-go" from a given list of steps.
        """
        idx_dones = [i for i, step in enumerate(steps) if step.done]
        try:
            assert len(idx_dones) == 1, (
                f"Invalid steps for one trajectory! "
                f"len({str(idx_dones)}) == {len(idx_dones)}, should be 1"
            )
        except AssertionError as e:
            logger.error("%s: %s", e.__class__.__name__, e)
            logger.error("idx_dones: %d", len(idx_dones))
            raise e

        if len(steps) > 0:
            try:
                assert steps[-1].done, f"Last step should be done! Dones at {idx_dones}"
            except AssertionError as e:
                logger.error("%s: %s", e.__class__.__name__, e)
                logger.error("steps[-1].done: %s", str(steps[-1].done))
        else:
            logger.error("No steps to get messages from")
            logger.error("steps: %s", steps)
            # pass  # breakpoint removed

        assert (full_rollout is True and timestep == -1) or not full_rollout, (
            "If full_rollout is True, timestep must be -1"
        )  # Enforce rules for sanity-checking

        timesteps = [step.timestep for step in steps]
        _step_idx = timesteps.index(timestep) if timestep != -1 else -1
        step = steps[_step_idx]
        return step.state + [step.action] + step.next_obs

    def get_messages_from_trajectory(
        self,
        trajectory: Trajectory,
        timestep: int = -1,
        full_rollout: bool = True,
    ) -> list[dict[str, str]]:
        """
        Get the messages from a trajectory (semi-alias for get_messages_from_steps)
        """
        return self.get_messages_from_steps(trajectory.episode_steps, timestep, full_rollout)

    def get_action_and_future_trajectory_from_steps(
        self,
        steps: list[EpisodeStep],
        action_step_idx: int = 0,
    ) -> tuple[dict[str, str], list[dict[str, str]]]:
        """
        Get the action and future trajectory from a list of steps

        Returns:
        - action (dict[str, str]): Action in the form {"role": "assistant", "content": <action>}
        - trajectory_to_go (list[dict[str, str]]): The future trajectory to go, in the form
          [state_t, action_t, next_obs_t, action_t+1, next_obs_t+1, ..., action_T, next_obs_T]
        """
        #  First check that there's only one done step at end of steps
        idx_dones = [i for i, step in enumerate(steps) if step.done]
        assert len(idx_dones) == 1, (
            f"Invalid steps for one trajectory! "
            f"len({str(idx_dones)}) == {len(idx_dones)}, should be 1"
        )
        assert steps[-1].done, f"Last step should be done! Dones at {idx_dones}"

        state: list[dict[str, str]] = steps[action_step_idx].state
        action: dict[str, str] = steps[action_step_idx].action
        # next_obs: list[dict[str, str]] = steps[timestep].next_obs
        trajectory_to_go = deepcopy(state)
        for _idx in range(action_step_idx, len(steps)):
            trajectory_to_go.append(steps[_idx].action)
            trajectory_to_go.extend(steps[_idx].next_obs)
        return action, trajectory_to_go

    def get_action_and_future_trajectory_from_trajectory(
        self,
        trajectory: Trajectory,
        action_step_idx: int = 0,
    ) -> tuple[dict[str, str], list[dict[str, str]]]:
        """
        Get the action and future trajectory from a trajectory
        """
        return self.get_action_and_future_trajectory_from_steps(
            trajectory.episode_steps,
            action_step_idx,
        )
