"""
Base class for Tinker-based generators
-> Implement methods as subclasses
"""

import asyncio
import logging
import sys
from typing import Any

from omegaconf import DictConfig
from tinker import SamplingParams
from tinker.types import ModelInput, SampleResponse
from tinker_cookbook.renderers import Renderer
from tinker_cookbook.utils import ml_log
from transformers import PreTrainedTokenizerBase

from critic_actor.environments import Environment, EnvironmentState, EnvironmentStepResult
from critic_actor.llm_handlers import TinkerCompleter
from critic_actor.llm_handlers.utils import get_actions
from critic_actor.llm_handlers.types import ActionFromLLM
from critic_actor.replay_buffer import ReplayBuffer
from critic_actor.replay_buffer.types import (
    EpisodeStep,
    MeanCenteredTrajectoryGroup,
    Trajectory,
    TrajectoryGroup,
)
from critic_actor.utils.display import RichTextStreamer, display_state_action_next_obs

from .utils import remove_prefix_messages

logger = logging.getLogger(__name__)

ROYGBIV = ["#FF0000", "#FF7F00", "#FFFF00", "#00FF00", "#0000FF", "#4B0082", "#9400D3"]
DEBUG_COLS = ["batch_id", "split", "try_step", "generation_id", "unique_data_sample_id"]


class TinkerGenerator:
    """
    Generate rollouts using Tinker SamplingClient
    """

    def __init__(
        self,
        llm: TinkerCompleter,
        hf_tokenizer: PreTrainedTokenizerBase,
        env: Environment,
        cfg: DictConfig,
        replay_buffer: ReplayBuffer,
        enable_thinking: bool | None = None,
        discount_factor: float | None = None,
        mean_center: bool = False,
        ml_logger: ml_log.Logger | None = None,
        name_or_identifier: str | None = None,
        streamer: bool = False,
        verbose: bool = False,
        last_generated_data_url: str | None = None,
        last_replay_buffer_path: str | None = None,
        debug: bool = False,
    ) -> None:
        self.llm = llm
        self.hf_tokenizer = hf_tokenizer
        self.env = env
        self.cfg = cfg
        self.replay_buffer = replay_buffer
        self.enable_thinking = enable_thinking

        self.discount_factor = discount_factor or cfg.get("discount_factor", 0.9)
        self.mean_center = mean_center  # mean-center the advantages

        self.ml_logger = ml_logger
        self.name_or_identifier = name_or_identifier
        self.run_url, self.run_cmd = self._init_identifiers()

        # Other identifiers that can be populated
        self.last_generated_data_url = last_generated_data_url
        self.last_replay_buffer_path = last_replay_buffer_path

        # Silly streaming (disabled for Tinker)
        self.streamer = (
            RichTextStreamer(self.hf_tokenizer, skip_prompt=True, skip_special_tokens=True)
            if streamer
            else None
        )
        self.verbose = verbose
        self.debug = debug

    def _init_identifiers(self) -> tuple[str | None, str | None]:
        """
        Initialize identifiers for the generator
        """
        run_url = self.ml_logger.get_logger_url() if self.ml_logger is not None else None
        run_cmd = self.cfg.get("run_cmd", " ".join(sys.argv))
        run_cmd = f"uv run {run_cmd}" if run_cmd else None
        return run_url, run_cmd

    def _get_trajectory_group(self, **kwargs: Any) -> TrajectoryGroup:
        """
        Return trajectory group class
        - Override in subclasses, e.g., to return MeanCenteredTrajectoryGroup
        """
        if self.mean_center:
            # Returns trajectory group where we compute advantages by:
            # 1. Computing mean-centered final rewards: final_reward - mean(final_rewards)
            # 2. Optionally apply step-wise discounting to these values
            return MeanCenteredTrajectoryGroup(**kwargs)
        return TrajectoryGroup(**kwargs)

    def _get_messages_from_state(
        self,
        state: EnvironmentState,
        default_context: list[dict[str, Any]] | None = None,
        **get_past_episode_steps_kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Get messages from the environment state, in the form of
        [{"role": <role>, "content": <content>}, ...]

        For multiple tries, as default behavior we build the context as:
        [system_prompt, default_context, prior_rollouts, *current_rollout]

        where *current_rollout is a list of the current:
        [prior_messages, last_model_response, new_messages]
        """
        # Initialize current rollout messages as state's
        # `prior observations + model's last response + environment new messages`
        # -> Hacky, but remove system prompt and default context if in prior messages
        default_context = default_context or state.default_context or []
        prior_messages = state.prior_messages or []
        prior_messages = remove_prefix_messages(prior_messages, default_context)
        messages = prior_messages + (state.model_response or []) + (state.new_messages or [])
        # -> Preprocess into {"role": <role>, "content": <content>} format
        #    - See `critic_actor.environments` classes for environment responses
        messages = [
            {"role": msg["role"], "content": msg["output"]}  # for consistency with
            if msg.get("type", "") == "function_call_output"  # OpenAI Responses API
            else msg
            for msg in messages
        ]
        # Return final messages list
        return [
            {"role": "system", "content": state.system_prompt},
            *default_context,
            *messages,
        ]

    async def do_single_rollout(
        self,
        llm: TinkerCompleter | None = None,
        env: Environment | None = None,
        hf_tokenizer: PreTrainedTokenizerBase | None = None,
        split: str = "train",
        batch_id: int = 0,
        sample_id: int = 0,
        generation_id: int = 0,
        try_step: int = 0,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Trajectory:
        """
        Run one full episode and return the resulting Trajectory.
        """
        llm = llm or self.llm
        env = env or self.env
        hf_tokenizer = hf_tokenizer or self.hf_tokenizer

        max_tokens = max_tokens or llm.max_tokens
        temperature = temperature or llm.temperature

        episode_steps: list[EpisodeStep] = []
        final_reward: float = 0.0
        truncated: bool = False

        state: EnvironmentState = await env.reset_async(
            sample_id=sample_id,
            generation_id=generation_id,
            try_step=try_step,
            batch_id=batch_id,
        )
        done = False

        while not done:
            state_messages = self._get_messages_from_state(
                state=state,
                split=split,
                timestep=state.timestep,
                try_step=state.try_step,
                replay_buffer=self.replay_buffer,
            )
            # Tokenize the current state for generation
            state_ids: list[int] = hf_tokenizer.apply_chat_template(
                conversation=state_messages,
                tools=state.tools,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
                tokenize=True,
                return_dict=False,
            )
            state_tinker_input: ModelInput = ModelInput.from_ints(state_ids)
            # Generate response (being explicit here instead of using the handler)
            renderer: Renderer = llm.renderer
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                stop=llm.stop_condition,  # tinker_cookbook.renders.Renderer
            )
            try:
                response: SampleResponse = await asyncio.wait_for(
                    llm.sampling_client.sample_async(
                        prompt=state_tinker_input,
                        num_samples=1,
                        sampling_params=sampling_params,
                    ),
                    timeout=120,  # 2 minute timeout per sample
                )
            except asyncio.TimeoutError:
                logger.error("Tinker sample_async timed out after 120s, retrying...")
                response: SampleResponse = await asyncio.wait_for(
                    llm.sampling_client.sample_async(
                        prompt=state_tinker_input,
                        num_samples=1,
                        sampling_params=sampling_params,
                    ),
                    timeout=120,
                )
            # Extract tokens and logprobs from the first (and only) sample
            sampled_tokens: list[int] = response.sequences[0].tokens
            sampled_logprobs: list[float] = response.sequences[0].logprobs
            assert sampled_logprobs is not None

            # Decode the response
            parsed_message, is_complete = renderer.parse_response(sampled_tokens)
            model_messages = [{"role": "assistant", "content": parsed_message["content"]}]
            parsed_actions: list[ActionFromLLM] = get_actions(model_messages)

            # Recompute logprobs over parsed actions
            state_action_ids: list[int] = hf_tokenizer.apply_chat_template(
                conversation=state_messages + model_messages,
                tools=state.tools,
                add_generation_prompt=False,
                enable_thinking=self.enable_thinking,
                tokenize=True,
                return_dict=False,
            )
            state_action_tinker_input = ModelInput.from_ints(state_action_ids)
            all_logprobs = await llm.compute_logprobs_async(state_action_tinker_input)
            # DEBUGGING
            assert len(all_logprobs) == len(state_action_ids) and all_logprobs[0] is None, (
                "tinker all_logprobs[0] should be None and match length of state_action_ids."
                f"\n- all_logprobs[0]: {all_logprobs[0]}"
                f"\n- len(all_logprobs): {len(all_logprobs)}"
                f"\n- len(state_action_ids): {len(state_action_ids)}"
            )
            act_token_len = len(state_action_ids) - len(state_ids)
            logprobs: list[float] = list(all_logprobs[-act_token_len:])

            # Step thru environment
            env_step_result: EnvironmentStepResult = await env.step_async(
                parsed_actions=parsed_actions,
                model_response=model_messages,
                current_state=state,
                current_messages=state_messages,
            )
            next_state = env_step_result.state
            reward = env_step_result.reward
            done = env_step_result.done
            truncated = env_step_result.truncated

            # 3. Save EpisodeStep
            next_obs = [
                {
                    "role": msg["role"],
                    "content": msg["output"] if msg.get("output", None) else msg["content"],
                }
                for msg in next_state.new_messages
            ]
            episode_steps.append(
                EpisodeStep(
                    state=state_messages,  # list[dict[str, str]]
                    action=model_messages[0],  # dict[str, str]
                    next_obs=next_obs,  # list[dict[str, str]]
                    tools=state.tools,
                    state_action_tokens=state_action_ids,
                    state_len=len(state_ids),
                    old_logprobs=logprobs,
                    temperature=temperature,
                    reward=reward,
                    done=done,
                    truncated=truncated,
                    timestep=state.timestep,
                    try_step=state.try_step,
                    batch_id=batch_id,
                    unique_data_sample_id=sample_id,
                    generation_id=generation_id,
                    split=split,
                    system_prompt=state.system_prompt,
                    task_prompt=state.task_prompt,
                    default_context=state.default_context,
                    is_complete=is_complete,
                )
            )
            if self.verbose:
                _header_text = (
                    f"(Method: Default) "
                    f"{split.title()} Split, Try {try_step}, Batch {batch_id},"
                    f" Sample {sample_id}, Generation {generation_id},"
                    f" Timestep {state.timestep}"
                )
                display_state_action_next_obs(
                    generator=self,
                    generation_id=generation_id,
                    state_messages=state_messages,
                    action_messages=model_messages,
                    next_obs_messages=next_obs,
                    tools=state.tools,
                    hf_tokenizer=hf_tokenizer,
                    header_text=_header_text,
                    group_rewards=[reward],  # final_reward
                    state=state,
                )
            # Transition to next state
            state = next_state
            final_reward = reward
            done = done or truncated

        return Trajectory(
            episode_steps=episode_steps,
            try_step=try_step,
            discount_factor=self.discount_factor,
            final_reward=final_reward,
        )

    async def do_group_rollout(
        self,
        num_return_sequences: int,
        **single_rollout_kwargs: Any,
    ) -> dict[str, list[TrajectoryGroup]]:
        """
        Generate a group of trajectories in the environment,
        and return a list of the trajectory group(s).

        By default, we should just return a singleton with 1 TrajectoryGroup. However, there may
        be cases for >1 TrajectoryGroups, e.g., if we're generating multiple actions per step,
        and we want advantages over each (state, action, next_obs) tuple across generations
        """
        trajectories_in_group: list[Trajectory] = await asyncio.gather(
            *[
                self.do_single_rollout(generation_id=gen_idx, **single_rollout_kwargs)
                for gen_idx in range(num_return_sequences)
            ]
        )
        all_trajectory_groups = [
            self._get_trajectory_group(
                trajectories=trajectories_in_group,
                discount_factor=self.discount_factor,
            )
        ]

        # Save replay buffer samples to disk
        if self.last_replay_buffer_path is not None:
            for trajectory_group in all_trajectory_groups:  # singleton
                for trajectory in trajectory_group.trajectories:
                    self.replay_buffer.add_trajectory(trajectory)
            self.replay_buffer.save_to_hf_dataset(self.last_replay_buffer_path)
            logger.info("Saved last replay buffer to %s", self.last_replay_buffer_path)

        # Sometimes we may want different trajectories to evaluate vs train on
        return {"policy": all_trajectory_groups}
