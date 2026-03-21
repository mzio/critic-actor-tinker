"""
Parent class PyTorch trainer for Hugging Face Transformers models
"""

import logging
import random
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable
from copy import deepcopy
from typing import Any

import numpy as np
import tinker
import torch
from omegaconf import DictConfig
from rich.console import Console
from tinker.types import LossFnType
from tinker_cookbook import model_info
from tinker_cookbook.renderers import Renderer, get_renderer
from tinker_cookbook.utils import ml_log
from transformers import PreTrainedTokenizerBase

from critic_actor.environments import Environment
from critic_actor.llm_handlers.tinker import StopCondition, TinkerCompleter
from critic_actor.replay_buffer import ReplayBuffer
from critic_actor.replay_buffer.types import Trajectory, TrajectoryGroup
from critic_actor.tinker_cookbook import (
    compute_full_batch_metrics_and_get_sampling_client,
    gather_with_progress,
    save_checkpoint_async,
    timed,
    train_step,
)
from critic_actor.generator import TinkerGenerator, get_generator_constructor

from .utils import display_metrics, hide_observations, is_better

logger = logging.getLogger(__name__)
console = Console()


class TinkerTrainer(ABC):
    """
    Parent class for Tinker trainers
    """

    def __init__(
        self,
        cfg: DictConfig,
        training_client: tinker.TrainingClient,
        service_client: tinker.ServiceClient,
        generator_cfg: DictConfig,
        replay_buffer: ReplayBuffer,
        env: Environment,
        eval_env: Environment,
        ml_logger: ml_log.Logger,
        hf_tokenizer: PreTrainedTokenizerBase | None = None,
        checkpoint_path: str | None = None,
        log_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.cfg = cfg
        self.training_client = training_client
        self.service_client = service_client

        # Initialize model info and renderer / tokenizers
        model_name = cfg.model_name or training_client.get_info().model_data.model_name
        hf_tokenizer = hf_tokenizer or training_client.get_tokenizer()
        # ^Same as tinker_cookbook.tokenizer_utils.get_tokenizer(cfg.model_name)?
        default_renderer_name = model_info.get_recommended_renderer_name(model_name)
        renderer_name = cfg.renderer_name or default_renderer_name
        self.renderer = get_renderer(renderer_name, hf_tokenizer)
        logger.info("Using renderer: %s", renderer_name)

        self.generator_cfg = generator_cfg
        self.replay_buffer = replay_buffer

        self.env = env
        self.eval_env = eval_env
        self.ml_logger = ml_logger
        self.hf_tokenizer = hf_tokenizer

        # If True, we hide observations other than the last one to avoid context blow-up
        # -> See hide_observations() in ../train.py for more details
        self.hide_observations = cfg.get("hide_observations", False)
        self.hidden_obs_content = cfg.get("hidden_obs_content", "...")

        # RL / Evaluation generator: does standard rollouts, see ../generator/base.py
        self.generator_constructor = self.get_generator_constructor(**self.generator_cfg)
        self.best_metric = 1e8 if "loss" in cfg.best_metric else -1e8
        self.best_metric_name = cfg.best_metric
        self.best_metric_step = -1

        # Cumulative cost tracking across all generator instances
        self._cumulative_cost_usd: float = 0.0

        self.run_name = cfg.run_name
        self.run_url = ml_logger.get_logger_url() if ml_logger is not None else None
        self.run_cmd = f"uv run {' '.join(sys.argv)}"

        # Logging and checkpointing
        self.checkpoint_path = checkpoint_path or cfg.checkpoint_path
        self.log_path = log_path or cfg.log_path

    def get_generator_constructor(self, **kwargs: Any) -> Callable[..., TinkerGenerator]:
        """
        Get a (partially initialized) TinkerGenerator constructor by name
        """
        return get_generator_constructor(
            cfg=self.cfg,
            ml_logger=self.ml_logger,
            replay_buffer=self.replay_buffer,
            **kwargs,
        )

    def _display_metrics(self, *args: Any, **kwargs: Any) -> None:
        """
        Display metrics in a table
        """
        display_metrics(*args, **kwargs)

    def _is_better(self, x: float, y: float, metric: str) -> bool:
        """
        Determine if x is better than y for a given metric
        """
        return is_better(x, y, metric)

    def _check_model_inputs(
        self,
        batch: dict[str, torch.Tensor],
        hf_tokenizer: PreTrainedTokenizerBase | None = None,
        cfg: DictConfig | None = None,
    ) -> None:
        """
        Sanity-check model inputs by rich printing them
        """
        hf_tokenizer = hf_tokenizer or self.hf_tokenizer
        cfg = cfg or self.cfg

        decoded_inputs = hf_tokenizer.batch_decode(batch["input_ids"][:, 1:])
        _labels = deepcopy(batch["labels"][:, 1:])
        _labels[_labels == -100] = 0  # -100 will cause tokenization errors
        decoded_labels = hf_tokenizer.batch_decode(_labels)

        advantages = batch.get("advantages")
        for idx, decoded_input in enumerate(decoded_inputs):
            console.print(f"[cyan]Input {idx}:\n{decoded_input}\n[/cyan]")
            console.print(f"[green]Label {idx}:\n{decoded_labels[idx]}\n[/green]")
            if advantages is not None:
                _adv = advantages[idx]
                _adv = _adv.tolist() if isinstance(_adv, torch.Tensor) else _adv
                console.print(f"[yellow]Advantage {idx}:\n{_adv}\n[/yellow]")
            console.print("=" * 100)
        # Keep run url and cmd in display
        console.print(f"[bold]Run url: [link={self.run_url}]{self.run_url}[/link][/bold]")
        console.print(f"[bold]Run cmd: [bright_cyan]{self.run_cmd}[/bright_cyan][/bold]")

    def maybe_hide_observations(
        self,
        messages: list[dict[str, str]],
        hidden_obs_content: str | None = None,
        first_obs_to_show: int = 2,  # e.g., to keep prompt
        last_obs_to_show: int = 1,  # e.g., to keep last observation
    ) -> list[dict[str, str]]:
        """
        Maybe hide past observations from messages
        """
        if not self.hide_observations:
            return messages

        hidden_obs_content = hidden_obs_content or self.hidden_obs_content
        return hide_observations(messages, hidden_obs_content, first_obs_to_show, last_obs_to_show)

    @abstractmethod
    async def prepare_minibatch(self, **kwargs: Any) -> tuple[list[tinker.Datum], dict[str, Any]]:
        """
        Prepare a minibatch of trajectories for training
        """
        raise NotImplementedError

    @abstractmethod
    def save_replay_buffer(
        self,
        replay_buffer: ReplayBuffer | None = None,
        best: bool = False,
    ) -> None:
        """
        Save replay buffer to disk
        """
        raise NotImplementedError

    @abstractmethod
    async def train(
        self,
        start_batch: int,
        end_batch: int,
        cfg: DictConfig | None = None,
        env: Environment | None = None,
        eval_env: Environment | None = None,
        eval_every: int | None = None,
        generator_constructor: Callable[..., TinkerGenerator] | None = None,
        # Other identifiers
        checkpoint_name: str | None = None,
        name_or_identifier: str | None = None,
        **kwargs: Any,
    ) -> str | None:
        """
        Implement entire training loop
        """
        raise NotImplementedError

    # Modified from https://github.com/thinking-machines-lab/tinker-cookbook/blob/22483a6b04400f79da13557a8229bc98b309b026/tinker_cookbook/rl/train.py#L941
    async def do_train_step_and_get_sampling_client(
        self,
        batch_idx: int,
        training_client: tinker.TrainingClient,
        data_D: list[tinker.Datum],
        prepare_minibatch_metrics: dict[str, Any] | None = None,
        loss_fn: LossFnType | None = None,
        checkpoint_name: str | None = None,
        mini_batch_size: int | None = None,
        num_substeps: int | None = None,
    ) -> tuple[tinker.SamplingClient, dict[str, Any]]:
        """
        Update LLM policy with new trajectories and return updated sampling client
        """
        cfg = self.cfg
        loss_fn = loss_fn or cfg.loss_fn

        mini_batch_size = mini_batch_size or cfg.mini_batch_size
        num_substeps = num_substeps or cfg.num_substeps

        metrics = {}
        # Minibatch created outside this function
        # data_D, prepare_minibatch_metrics = await self.prepare_minibatch(
        #     new_trajectories=new_trajectories,
        #     service_client=service_client,
        #     model_name=cfg.model_name,
        #     kl_penalty_coef=cfg.kl_penalty_coef,
        #     kl_discount_factor=cfg.kl_discount_factor,
        # )
        metrics.update(prepare_minibatch_metrics or {})

        # Resample from data_D to determine actual training set
        if mini_batch_size and num_substeps is None:
            num_substeps = len(data_D) // mini_batch_size
        elif num_substeps:
            if mini_batch_size:
                # Potentially supersample data_D to hit mini_batch_size * num_substeps
                data_D = data_D[: mini_batch_size * num_substeps]
                while len(data_D) < mini_batch_size * num_substeps:
                    data_D.extend(random.sample(data_D, k=len(data_D)))
                data_D = data_D[: mini_batch_size * num_substeps]
        else:
            raise ValueError("Either mini_batch_size or num_substeps must be specified")
        # Randomly subsample training data if not evenly divisible by num_substeps (# of mini-batches)
        # -> Tinker requires this: https://tinker-docs.thinkingmachines.ai/rl/rl-hyperparams#multiple-updates-per-sampling-iteration
        if len(data_D) % num_substeps != 0:
            new_batch_size = (len(data_D) // num_substeps) * num_substeps
            data_D = random.sample(data_D, new_batch_size)
        random.shuffle(data_D)  # Ensure random ordering of mini-batches

        with timed("train", metrics):
            training_logprobs_D = await train_step(
                data_D,
                training_client,
                cfg.learning_rate,
                num_substeps,
                loss_fn,
            )
        (
            sampling_client,
            full_batch_metrics,
        ) = await compute_full_batch_metrics_and_get_sampling_client(
            training_client,
            batch_idx + 1,  # NOTE: saving the checkpoint as the i + 1 step
            data_D,
            training_logprobs_D,
            cfg.log_path,
            cfg.save_every,
            cfg.compute_post_kl,
            checkpoint_name=checkpoint_name,
            do_compute_kl=loss_fn != "cross_entropy",
        )
        metrics.update(full_batch_metrics)
        return sampling_client, metrics

    async def run_rollouts(
        self,
        batch_id: int,
        sampling_client: tinker.SamplingClient,
        env: Environment,
        split: str,
        cfg: DictConfig | None = None,
        renderer: Renderer | None = None,
        hf_tokenizer: PreTrainedTokenizerBase | None = None,
        generator_constructor: Callable[..., TinkerGenerator] | None = None,
        checkpoint_name: str | None = None,
        num_tries: int = 1,
        start_idx: int = 0,
        tasks_per_update: int | None = None,
        name_or_identifier: str | None = None,
        pbar_position: int = 0,
        eval_trajectory_key: str | None = None,
        # Overrides for generation
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop_condition: StopCondition | None = None,
    ) -> tuple[dict[str, Any], dict[str, list[Trajectory]]]:
        """
        Run rollouts for a single batch
        - Implemented here because we could also do sampling-based evals during 
          an SFT training loop (behavioral cloning evaluation)
        """

        # Load potential defaults tied to the Trainer object
        cfg = cfg or self.cfg
        renderer = renderer or self.renderer
        hf_tokenizer = hf_tokenizer or self.hf_tokenizer
        generator_constructor = generator_constructor or self.generator_constructor
        max_tokens = max_tokens or cfg.max_tokens
        temperature = temperature or cfg.temperature
        stop_condition = stop_condition or cfg.stop_condition

        env.split = split  # Specify task split

        # Initialize TinkerCompleter from provided sampling client weights
        tinker_completer = TinkerCompleter(
            sampling_client=sampling_client,
            renderer=renderer,
            max_tokens=max_tokens,
            temperature=temperature,
            stop_condition=stop_condition,
            hf_tokenizer=hf_tokenizer,
            **cfg.tool_call_kwargs,
        )
        # Constructs a TinkerGenerator object (e.g., default, PGIC, etc.)
        tinker_generator: TinkerGenerator = generator_constructor(
            llm=tinker_completer,
            env=env,
            hf_tokenizer=hf_tokenizer,
            name_or_identifier=name_or_identifier,
        )

        batch_size = tasks_per_update or len(env)  # len(env) is the number of tasks or problems
        num_return_sequences = cfg.group_size if split == "train" else cfg.eval_group_size

        all_eval_metrics: dict[str, list[float]] = {}
        keys_for_correct: list[str] = []
        eval_metric_keys = [
            "final_reward",
            "first_return",
            "action_prob",
            "last_state_len",
            "timesteps",
            "correct",
            "total",
        ]
        # Store new trajectories to return
        new_trajectories: dict[str, list[Trajectory]] = {}

        for try_idx in range(num_tries):
            all_trajectory_groups: list[
                dict[str, list[TrajectoryGroup]]
            ] = await gather_with_progress(
                (
                    tinker_generator.do_group_rollout(
                        num_return_sequences=num_return_sequences,
                        split=split,
                        batch_id=batch_id,
                        sample_id=sample_idx,
                        try_step=try_idx,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    for sample_idx in range(start_idx, start_idx + batch_size)
                ),
                colour="blue" if split == "train" else "magenta",
                desc=(
                    f"Generating {batch_size * num_return_sequences} {split.upper()} rollouts "
                    f"({batch_size} tasks, {num_return_sequences} per task)"
                ),
                position=pbar_position,
            )
            # Filter out None results (timed-out tasks)
            all_trajectory_groups = [tg for tg in all_trajectory_groups if tg is not None]
            if not all_trajectory_groups:
                logger.warning("All tasks timed out in this batch, skipping")
                continue

            # Save metrics and samples
            _rollout_types = all_trajectory_groups[0].keys()  # e.g., ["policy", "icl"]
            _metric_prefix = f"{checkpoint_name}_{split}" if checkpoint_name is not None else split

            # MZ 02/11/26 TODO: Make this nesting less heinous...
            eval_type = eval_trajectory_key or cfg.get("eval_trajectory_key", "policy")
            eval_type = "policy" if eval_type not in _rollout_types else eval_type
            for _type in _rollout_types:
                # Initialize lists to save each rollout type
                if _type not in new_trajectories:
                    new_trajectories[_type] = []

                for trajectory_groups in all_trajectory_groups:  # list of list of trajectory groups
                    for traj_group in trajectory_groups[_type]:  # len(trajectory_groups) usually 1,
                        for (
                            trajectory
                        ) in traj_group.trajectories:  # can be >1, e.g., if step-wise adv
                            # Only store metrics for the specified trajectory group (default "policy")
                            if _type == eval_type:
                                for metric_key in eval_metric_keys:
                                    all_metric_key = f"{_metric_prefix}/try_{try_idx}/{metric_key}"
                                    if metric_key == "correct":
                                        keys_for_correct.append(all_metric_key)
                                    if all_metric_key not in all_eval_metrics:
                                        all_eval_metrics[all_metric_key] = []
                                    val = getattr(trajectory, metric_key, 1)  # 1 for total samples
                                    all_eval_metrics[all_metric_key].append(val)
                            # Add trajectory to list of new trajectories
                            new_trajectories[_type].append(trajectory)

                            if _type == "policy":
                                # (potentially redundant with prepare_minibatch)
                                self.replay_buffer.add_trajectory(trajectory)
                # Save replay buffer after each try
                self.save_replay_buffer(replay_buffer=self.replay_buffer, best=False)

        final_metrics = {}  # return these metrics for the batch
        # 1. Compute aggregate metrics
        for k, v in all_eval_metrics.items():
            if "correct" in k or "total" in k:
                final_metrics[k] = np.sum(v).item()  # convert to float for json.dumps
            else:
                final_metrics[k] = np.mean(v).item()
            final_metrics[f"{k}_std"] = np.std(v).item()
            final_metrics[f"{k}_max"] = np.max(v).item()
        # 2. Add accuracy metrics
        for k in keys_for_correct:
            total_v = final_metrics[k.replace("correct", "total")]
            final_metrics[k.replace("correct", "accuracy")] = final_metrics[k] / total_v

        # Pull cost metrics from generator if available
        if hasattr(tinker_generator, "get_cost_metrics"):
            cost_metrics = tinker_generator.get_cost_metrics()
            self._cumulative_cost_usd += cost_metrics.get("cost/batch_usd", 0)
            cost_metrics["cost/cumulative_usd"] = self._cumulative_cost_usd
            final_metrics.update(cost_metrics)

        return final_metrics, new_trajectories

    async def save_best_checkpoint_and_replay_buffer(
        self,
        eval_rollout_metrics: dict[str, Any],
        eval_split: str,
        batch_idx: int,
        metrics: dict[str, Any],
        checkpoint_name: str | None = None,
        training_client: tinker.TrainingClient | None = None,
        replay_buffer: ReplayBuffer | None = None,
    ) -> None:
        """
        Save best checkpoint
        """
        training_client = training_client or self.training_client
        replay_buffer = replay_buffer or self.replay_buffer

        best_metric_key = [k for k in eval_rollout_metrics if self.best_metric_name in k][0]
        last_metric = eval_rollout_metrics[best_metric_key]
        best_ckpt_name = (
            f"{batch_idx:06d}_best"
            if checkpoint_name is None
            else f"{checkpoint_name}_{batch_idx:06d}_best"
        )
        if self._is_better(last_metric, self.best_metric, self.best_metric_name):
            self.best_metric = last_metric
            self.best_metric_step = batch_idx

            path_dict = await save_checkpoint_async(
                training_client=training_client,
                name=best_ckpt_name,
                log_path=self.cfg.log_path,
                loop_state={"batch": batch_idx},
                kind="both",
            )
            best_sampling_client_path = path_dict["sampler_path"]
            logger.info("Saved best sampling client to %s", best_sampling_client_path)
            logger.info("Saved best state (weights) to %s", path_dict["state_path"])
            logger.info(
                "Updated best %s to %f at batch %d",
                self.best_metric_name,
                self.best_metric,
                batch_idx,
            )
            metrics.update(
                {
                    f"{eval_split}/{self.best_metric_name}": last_metric,
                    f"{eval_split}/{self.best_metric_name}_best": self.best_metric,
                    f"{eval_split}/{self.best_metric_name}_best_step": self.best_metric_step,
                    f"{eval_split}/{self.best_metric_name}_best_sampling_client_path": best_sampling_client_path,
                    f"{eval_split}/{self.best_metric_name}_best_state_path": path_dict[
                        "state_path"
                    ],
                }
            )
            self.save_replay_buffer(replay_buffer=replay_buffer, best=True)
