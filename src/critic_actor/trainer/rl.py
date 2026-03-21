"""
Tinker SFT Trainer
"""

import logging
import time
from collections.abc import Callable
from os.path import join
from typing import Any

import tinker
import torch
from datasets.arrow_writer import SchemaInferenceError
from omegaconf import DictConfig
from tinker.types import Datum, ModelInput, TensorData
from tinker_cookbook import model_info, renderers

from critic_actor.environments import Environment
from critic_actor.replay_buffer import ReplayBuffer, Trajectory
from critic_actor.tinker_cookbook import (
    incorporate_kl_penalty,
    save_checkpoint_and_get_sampling_client,
    timed,
)
from critic_actor.generator import TinkerGenerator


from .base import TinkerTrainer

logger = logging.getLogger(__name__)


class RlTinkerTrainer(TinkerTrainer):
    """
    Trainer for reinforcement learning (via policy gradient) with Tinker
    """

    def __init__(
        self,
        cfg: DictConfig,
        training_client: tinker.TrainingClient,
        service_client: tinker.ServiceClient,
        generator_cfg: DictConfig,
        replay_buffer: ReplayBuffer,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            cfg=cfg,
            training_client=training_client,
            service_client=service_client,
            generator_cfg=generator_cfg,
            replay_buffer=replay_buffer,
            **kwargs,
        )
        self.replay_buffer = replay_buffer  # Just to be explicit
        self.best_replay_buffer_path = join(cfg.checkpoint_path, "replay_buffer_best")
        self.last_replay_buffer_path = join(cfg.checkpoint_path, "replay_buffer")
        
        # Re-initialize generator constructor to include claude agent and last replay buffer path
        self.generator_constructor = self.get_generator_constructor(
            last_replay_buffer_path=self.last_replay_buffer_path,
            **self.generator_cfg,
        )

    def save_replay_buffer(
        self,
        replay_buffer: ReplayBuffer | None = None,
        best: bool = False,
    ) -> None:
        """
        Save replay buffer to disk
        """
        replay_buffer = replay_buffer or self.replay_buffer
        assert replay_buffer is not None, "Replay buffer is not set"
        save_path = self.best_replay_buffer_path if best else self.last_replay_buffer_path
        try:
            replay_buffer.save_to_hf_dataset(save_path)
            logger.info("Saved replay buffer to %s", save_path)
        except SchemaInferenceError:
            logger.warning("Failed to save replay buffer to %s\nIs replay buffer empty?", save_path)

    async def prepare_minibatch(self, **kwargs: Any) -> tuple[list[tinker.Datum], dict[str, Any]]:
        """
        Prepare a minibatch of trajectories for SFT training
        """
        return await self._prepare_minibatch(**kwargs)

    async def _prepare_minibatch(
        self,
        new_trajectories: list[Trajectory],
        service_client: tinker.ServiceClient | None = None,  # self.service_client
        model_name: str | None = None,  # cfg.model_name,
        kl_penalty_coef: float | None = None,  # cfg.kl_penalty_coef
        kl_discount_factor: float | None = None,  # cfg.kl_discount_factor
    ) -> tuple[list[tinker.Datum], dict[str, Any]]:
        """
        Prepare a minibatch of trajectories for RL training
        """
        metrics = {}

        service_client = service_client or self.service_client
        model_name = model_name or self.cfg.model_name
        kl_penalty_coef = kl_penalty_coef or self.cfg.kl_penalty_coef
        kl_discount_factor = kl_discount_factor or self.cfg.kl_discount_factor

        with timed("assemble_training_data", metrics):
            data_D: list[Datum] = []
            metadata_D: list[dict[str, int]] = []
            for trajectory in new_trajectories:
                _all_trajectory_episode_steps = (
                    trajectory.episode_steps
                    + trajectory.post_rollout_episode_steps
                )
                for episode_step in _all_trajectory_episode_steps:
                    # sa_input_ids = episode_step.state_action_tokens
                    advantage = episode_step.advantage

                    if (  # hard-coded hack for the cria case
                        episode_step.other_state_action_tokens is not None and advantage < 0
                    ):
                        all_sa_input_ids = episode_step.other_state_action_tokens
                        target_advantage = [-advantage] * len(all_sa_input_ids)
                    else:
                        all_sa_input_ids = [episode_step.state_action_tokens]
                        target_advantage = [advantage] * len(all_sa_input_ids)

                    for _idx, sa_input_ids in enumerate(all_sa_input_ids):
                        sa_advantage = target_advantage[_idx]
                        input_tokens = sa_input_ids[:-1]
                        target_tokens = sa_input_ids[1:]
                        act_logprobs = (
                            episode_step.old_logprobs   # bugged here? because should match other_state_action tokens then
                        )  # logprob for predicting each action token

                        target_state_len = episode_step.state_len - 1
                        target_advantage = sa_advantage
                        # ^for now assume the same advantage for all tokens

                        padded_logprobs = [0.0] * target_state_len + act_logprobs
                        padded_advantages = [0.0] * target_state_len + [target_advantage] * len(
                            act_logprobs
                        )
                        padded_mask = [0.0] * target_state_len + [1.0] * len(act_logprobs)

                        try:
                            assert (
                                len(input_tokens)
                                == len(padded_logprobs)
                                == len(padded_advantages)
                                == len(target_tokens)
                            )
                        except AssertionError as e:
                            logger.error(
                                "Length mismatch:"
                                "\n\tinput=%d"
                                "\n\tlogprobs=%d"
                                "\n\tadvantages=%d"
                                "\n\ttargets=%d",
                                len(input_tokens),
                                len(padded_logprobs),
                                len(padded_advantages),
                                len(target_tokens),
                            )
                            breakpoint()
                            raise e

                        metadata_D.append(
                            {
                                "sample_id": episode_step.unique_data_sample_id,
                                "generation_id": episode_step.generation_id,
                            }
                        )
                        data_D.append(
                            Datum(
                                model_input=ModelInput.from_ints(input_tokens),
                                loss_fn_inputs={
                                    "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                                    "logprobs": TensorData.from_torch(torch.tensor(padded_logprobs)),
                                    "advantages": TensorData.from_torch(
                                        torch.tensor(padded_advantages)
                                    ),
                                    "mask": TensorData.from_torch(torch.tensor(padded_mask)),  # for KL
                                },
                            )
                        )

        # Incorporate KL penalty if configured
        # - Copied from https://github.com/thinking-machines-lab/tinker-cookbook/blob/22483a6b04400f79da13557a8229bc98b309b026/tinker_cookbook/rl/train.py#L763
        if kl_penalty_coef > 0:
            with timed("kl_vs_base", metrics):
                kl_penalty_metrics = await incorporate_kl_penalty(
                    data_D,
                    service_client.create_sampling_client(base_model=model_name),
                    # ^^^ TODO: replace with the model we load, if relevant
                    kl_penalty_coef,
                    kl_discount_factor,
                )
            metrics.update(kl_penalty_metrics)

        return data_D, metrics

    async def _do_train_step_and_get_sampling_client(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[tinker.SamplingClient, dict[str, Any]]:
        """
        Update LLM policy with new trajectories and return updated sampling client
        -> Weak wrapper around parent class method
        """
        return await super().do_train_step_and_get_sampling_client(*args, **kwargs)

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
        Implement entire SFT training loop
        """

        cfg = cfg or self.cfg
        env = env or self.env
        eval_env = eval_env or self.eval_env
        eval_every = eval_every or cfg.eval_every
        eval_start_batch = getattr(cfg, "eval_start_batch", 0) or 0
        generator_constructor = generator_constructor or self.generator_constructor

        best_sampling_client_path = ""  # Path to the best sampling client

        # Initial sampling client
        sampling_client, _ = await save_checkpoint_and_get_sampling_client(
            training_client=self.training_client,
            i_batch=start_batch,
            log_path=cfg.log_path,
            save_every=cfg.save_every,
            start_batch=start_batch,
            checkpoint_name=checkpoint_name,
        )

        model_name = cfg.model_name or self.training_client.get_info().model_data.model_name
        hf_tokenizer = self.hf_tokenizer or self.training_client.get_tokenizer()
        # ^Same as tinker_cookbook.tokenizer_utils.get_tokenizer(cfg.model_name)?
        renderer_name = cfg.renderer_name or model_info.get_recommended_renderer_name(model_name)
        renderer = renderers.get_renderer(renderer_name, hf_tokenizer)
        logger.info("Using renderer: %s", renderer_name)

        num_batches = end_batch - start_batch
        wen_shuffle = len(env.datasets["train"])

        for batch_idx in range(start_batch, end_batch):
            metrics = {
                "progress/batch": batch_idx,
                "optim/lr": cfg.learning_rate,
                "progress/done_frac": (batch_idx + 1) / num_batches,
            }
            t_start = time.time()

            # 2. Run evaluations
            if (
                eval_env is not None
                and eval_every > 0
                and batch_idx % eval_every == 0
                and batch_idx >= eval_start_batch
            ):
                eval_num_tries = cfg.eval_num_tries or eval_env.num_tries
                for eval_split in getattr(eval_env, "eval_splits", ["eval"]):
                    with timed("run_evals", metrics):
                        eval_env.split = eval_split
                        eval_rollout_metrics, _ = await self.run_rollouts(
                            batch_id=batch_idx,
                            sampling_client=sampling_client,
                            env=eval_env,
                            split=eval_split,
                            cfg=cfg,
                            renderer=renderer,
                            hf_tokenizer=hf_tokenizer,
                            generator_constructor=generator_constructor,
                            checkpoint_name=checkpoint_name,
                            num_tries=eval_num_tries,
                            start_idx=0,
                            tasks_per_update=len(eval_env),
                            name_or_identifier=name_or_identifier,
                            eval_trajectory_key="policy",
                        )
                        metrics.update(eval_rollout_metrics)
                        # Pull per-turn judge metrics from env (reward per turn, scores per turn)
                        if hasattr(eval_env, "get_judge_metrics"):
                            metrics.update(eval_env.get_judge_metrics(
                                prefix=f"{eval_split}"
                            ))

                        display_title = (
                            f"Rollout {eval_split.capitalize()} Metrics, Step {batch_idx}"
                        )
                        self._display_metrics(
                            eval_rollout_metrics,
                            title=display_title,
                            style="bright_yellow",
                        )

                    # Save best checkpoints and replay buffer
                    await self.save_best_checkpoint_and_replay_buffer(
                        eval_rollout_metrics=eval_rollout_metrics,
                        eval_split=eval_split,
                        batch_idx=batch_idx,
                        metrics=metrics,
                        checkpoint_name=checkpoint_name,
                        training_client=self.training_client,
                        replay_buffer=self.replay_buffer,
                    )

            # 1. Sample rollouts for training
            start_idx = batch_idx * cfg.batch_size
            tasks_per_update = cfg.batch_size
            env.split = "train"
            if start_idx + tasks_per_update > wen_shuffle:
                env.shuffle(split="train")
                wen_shuffle += len(env.datasets["train"])

            with timed("sample_rollouts", metrics):
                train_rollout_metrics, new_trajectories = await self.run_rollouts(
                    batch_id=batch_idx,
                    sampling_client=sampling_client,
                    env=env,
                    split="train",
                    generator_constructor=self.generator_constructor,
                    checkpoint_name=checkpoint_name,
                    num_tries=cfg.num_tries,
                    start_idx=start_idx,
                    tasks_per_update=tasks_per_update,
                    name_or_identifier=name_or_identifier,
                    eval_trajectory_key="policy",
                )
            metrics.update(train_rollout_metrics)
            # Pull per-turn judge metrics from train env
            if hasattr(env, "get_judge_metrics"):
                metrics.update(env.get_judge_metrics(prefix="train"))


            # 1.1. Collect trajectories for training policy update
            # _new_trajectories: list[Trajectory] = []
            # for trajectory in new_trajectories["policy"]:
            #     _new_trajectories.append(trajectory)
            #     # self.replay_buffer.add_trajectory(trajectory)  # (potentially redundant with prepare_minibatch)
            # # self.save_replay_buffer(replay_buffer=self.replay_buffer, best=False)
            # new_trajectories = _new_trajectories

            # 2. Update policy LLM with generated rollouts
            data_D, prepare_minibatch_metrics = await self.prepare_minibatch(
                new_trajectories=new_trajectories["policy"],
                service_client=self.service_client,
                model_name=cfg.model_name,
                kl_penalty_coef=cfg.kl_penalty_coef,
                kl_discount_factor=cfg.kl_discount_factor,
            )
            sampling_client, update_metrics = await self._do_train_step_and_get_sampling_client(
                batch_idx=batch_idx,
                training_client=self.training_client,
                data_D=data_D,
                prepare_minibatch_metrics=prepare_minibatch_metrics,
                loss_fn="importance_sampling",
                checkpoint_name=checkpoint_name,
            )

            # Log metrics
            metrics.update(update_metrics)
            metrics["time/total"] = time.time() - t_start
            self.ml_logger.log_metrics(metrics, step=batch_idx)

        return best_sampling_client_path
