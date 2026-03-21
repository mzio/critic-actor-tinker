"""
Chat OpenAI Generator

Same architecture as ChatClaudeGenerator but uses OpenAI Responses API
(e.g. GPT-5-mini) instead of Claude Agent SDK for the action model.

Flow:
1. Policy LLM (Tinker) generates a chat message (tip / reflection)
2. Chat message is injected into the OpenAI prompt as context
3. OpenAI model produces a tool-call action informed by the chat message
4. Environment steps with OpenAI's action → reward
5. Training signal: policy LLM's generated chat tokens, reward from env.step
"""

import asyncio
import json
import logging
from copy import deepcopy
from typing import Any

from rich import print as rich_print

from tinker import SamplingParams
from tinker.types import ModelInput, SampleResponse
from transformers import PreTrainedTokenizerBase

from critic_actor.environments import Environment, EnvironmentState, EnvironmentStepResult
from critic_actor.llm_handlers import AsyncOpenAIResponsesLLM, TinkerCompleter
from critic_actor.llm_handlers.utils import get_actions
from critic_actor.replay_buffer.types import (
    Trajectory,
    EpisodeStep,
)
from critic_actor.utils.display import display_state_action_next_obs

from ..base import TinkerGenerator
from ..cria_claude.utils import convert_tools_dict_to_str

# Reuse policy-side prompts from chat_claude
from ..chat_claude.prompts import (
    POLICY_CHAT_SYSTEM_PROMPT_TEMPLATE,
    CLAUDE_SYSTEM_PROMPT_TEMPLATE_HANDOFF,
)
from ..chat_claude.generator import POLICY_HANDOFF_INSTRUCTION, CLAUDE_HANDOFF_PREFIX_CHAT

logger = logging.getLogger(__name__)


class ChatOpenAIGenerator(TinkerGenerator):
    """
    Chat OpenAI Generator: policy LLM produces a chat message (tip / reflection),
    which is injected into an OpenAI model's prompt. The OpenAI model acts using
    native tool calling, and the reward trains the policy on its chat generation.
    """

    def __init__(
        self,
        prompt_template_name: str = "handoff",
        # OpenAI options
        openai_model: str = "gpt-5-mini",
        openai_base_url: str | None = None,
        openai_generation_config: dict[str, Any] | None = None,
        # Tinker options
        tinker_timeout: int = 300,
        no_cria_prompt: bool = False,
        continue_prompt: str | None = "Okay, let me think about",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.prompt_template_name = prompt_template_name
        self.no_cria_prompt = no_cria_prompt
        self.continue_prompt = continue_prompt
        self.tinker_timeout = tinker_timeout

        # OpenAI Responses API client
        self.openai_model = openai_model
        self.openai_llm = AsyncOpenAIResponsesLLM(
            model=openai_model,
            base_url=openai_base_url,
            generation_config=openai_generation_config or {},
        )

        # Semaphore to limit concurrent OpenAI requests (avoid TPM rate limits)
        self._openai_semaphore = asyncio.Semaphore(4)

        # Cumulative cost tracking
        self._cumulative_cost_usd: float = 0.0
        self._batch_cost_usd: float = 0.0

    def get_cost_metrics(self) -> dict[str, float]:
        """Return cost metrics and reset batch cost."""
        metrics = {
            "cost/batch_usd": self._batch_cost_usd,
            "cost/cumulative_usd": self._cumulative_cost_usd,
            "cost/openai_prompt_tokens": self.openai_llm.prompt_tokens,
            "cost/openai_completion_tokens": self.openai_llm.completion_tokens,
        }
        self._batch_cost_usd = 0.0
        return metrics

    def _get_messages_from_state(
        self,
        state: EnvironmentState,
        include_system_prompt: bool = False,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Get messages from the environment state.
        Returns (all_messages, new_messages).
        """
        new_messages = [
            {"role": msg["role"], "content": msg["output"]}
            if msg.get("type", "") == "function_call_output"
            else msg
            for msg in state.new_messages
        ]
        model_response: list[dict[str, str]] = state.model_response or []
        all_messages = (
            (state.prior_messages or []) + model_response + new_messages
        )
        if all_messages and all_messages[0].get("role", "") == "system" and not include_system_prompt:
            all_messages = all_messages[1:]
        return all_messages, new_messages

    # ------------------------------------------------------------------
    # Policy LLM generation (same as chat_claude)
    # ------------------------------------------------------------------
    async def _generate_policy_chat(
        self,
        llm: TinkerCompleter,
        hf_tokenizer: PreTrainedTokenizerBase,
        state: EnvironmentState,
        all_messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        tinker_timeout: int | None = None,
        continue_prompt: str | None = None,
    ) -> tuple[str, list[int], list[int], list[float]]:
        """
        Have the policy LLM generate a chat message (tip / reflection).

        Returns:
            model_text: the decoded model message
            state_ids: tokenized state (prefix)
            state_action_ids: tokenized state + model generation
            action_logprobs: log-probabilities for generated tokens
        """
        tinker_timeout = tinker_timeout or self.tinker_timeout
        continue_prompt = continue_prompt or self.continue_prompt

        all_messages = deepcopy(all_messages)
        if continue_prompt:
            all_messages.append({"role": "assistant", "content": continue_prompt})
        state_ids: list[int] = hf_tokenizer.apply_chat_template(
            conversation=all_messages,
            add_generation_prompt=continue_prompt is None,
            continue_final_message=continue_prompt is not None,
            enable_thinking=self.enable_thinking,
            tokenize=True,
            return_dict=False,
        )
        state_tinker_input = ModelInput.from_ints(state_ids)

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            stop=llm.stop_condition,
        )

        acceptable_generation = False
        while not acceptable_generation:
            try:
                response: SampleResponse = await asyncio.wait_for(
                    llm.sampling_client.sample_async(
                        prompt=state_tinker_input,
                        num_samples=1,
                        sampling_params=sampling_params,
                    ),
                    timeout=tinker_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning("Policy LLM generation timed out, retrying...")
                response = await asyncio.wait_for(
                    llm.sampling_client.sample_async(
                        prompt=state_tinker_input,
                        num_samples=1,
                        sampling_params=sampling_params,
                    ),
                    timeout=tinker_timeout,
                )

            sampled_tokens: list[int] = response.sequences[0].tokens
            sampled_logprobs: list[float] = response.sequences[0].logprobs
            assert sampled_logprobs is not None

            parsed_message, is_complete = llm.renderer.parse_response(sampled_tokens)
            model_text = parsed_message["content"]
            if isinstance(model_text, list):
                model_text = "\n".join(
                    p["text"] for p in model_text if p.get("type") == "text"
                )

            assert "<final_response>" in model_text and "</final_response>" in model_text, (
                "Expected <final_response> tags in policy model generation"
            )
            acceptable_generation = True

        model_text = continue_prompt + model_text if continue_prompt else model_text
        model_message = [{"role": "assistant", "content": model_text}]
        state_action_ids: list[int] = hf_tokenizer.apply_chat_template(
            conversation=all_messages + model_message,
            tools=state.tools,
            add_generation_prompt=False,
            enable_thinking=self.enable_thinking,
            tokenize=True,
            return_dict=False,
        )
        state_action_tinker_input = ModelInput.from_ints(state_action_ids)

        all_logprobs = await llm.compute_logprobs_async(state_action_tinker_input)
        action_token_len = len(state_action_ids) - len(state_ids)
        action_logprobs: list[float] = list(all_logprobs[-action_token_len:])

        return model_text, state_ids, state_action_ids, action_logprobs

    # ------------------------------------------------------------------
    # OpenAI action from policy chat
    # ------------------------------------------------------------------
    async def _get_openai_action(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        policy_text: str | None,
        sample_id: int,
    ) -> list[dict[str, str]]:
        """
        Query OpenAI with the conversation context plus the policy's chat
        message. Uses native tool calling via the Responses API.

        Returns model_messages (list of assistant message dicts).
        """
        # Inject policy handoff note into the messages
        messages = deepcopy(messages)
        # hack, but convert all tool responses to user messages
        try:
            messages = [
                {
                    "role": "user" if msg["role"] == "tool" else msg["role"],
                    "content": (
                        f"[Tool Response]\n{msg["content"]}"
                        if msg["role"] == "tool"
                        else msg["content"]
                    ),
                }
                for msg in messages if msg.get("role", None) is not None
            ]
        except Exception as e:
            print(messages)
            breakpoint()
        if policy_text is not None:
            handoff_content = CLAUDE_HANDOFF_PREFIX_CHAT.format(policy_text=policy_text)
            # messages.append({"role": "user", "content": handoff_content})
            messages[-1]["content"] = f"{messages[-1]['content']}\n\n{handoff_content}"

        _important_instructions = (
            "# IMPORTANT\n\n"
            "You cannot call tools yourself. You must only suggest tool calls as messages"
            "to the user. These must be in the described format. The user can also only"
            "consider a single tool call at a time."
        )
        system_prompt = f"{system_prompt}\n\n{_important_instructions}"

        num_attempts = 0
        max_attempts = 10
        while num_attempts < max_attempts:
            async with self._openai_semaphore:
                try:
                    responses = await self.openai_llm.sample_async(
                        system_prompt=system_prompt,
                        messages=messages,
                        tools=None,  # [],  # tools,
                        max_new_tokens=1024,
                        num_return_sequences=1,
                    )
                    response = responses[0]
                    if response is None:
                        wait = min(2 ** num_attempts, 60)
                        logger.warning(
                            'OpenAI returned None (rate limit?), attempt %d/%d, waiting %.1fs',
                            num_attempts + 1, max_attempts, wait,
                        )
                        num_attempts += 1
                        await asyncio.sleep(wait)
                        continue

                    actions = self.openai_llm.get_actions(response)
                    if not actions:
                        raise ValueError('No actions returned from OpenAI response')

                    content_parts = []
                    for action in actions:
                        if action.type == 'reasoning':
                            content_parts.append(action.text)
                        elif action.type == 'message':
                            content_parts.append(action.text)
                        elif action.type == 'function_call':
                            content_parts.append(action.text)

                    content = "\n\n".join(content_parts)
                    return [{'role': 'assistant', 'content': content}]

                except Exception as e:
                    logger.error('OpenAI action error (%s): %s', e.__class__.__name__, e)
                    if 'rate_limit' in str(e).lower() or '429' in str(e):
                        wait = min(2 ** num_attempts, 60)
                        logger.warning('Rate limit hit, waiting %.1fs before retry', wait)
                        await asyncio.sleep(wait)
                    num_attempts += 1

        raise RuntimeError(f'Failed to get valid OpenAI action after {max_attempts} attempts')

    # ------------------------------------------------------------------
    # Main rollout
    # ------------------------------------------------------------------
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
        """Run one full episode and return the resulting Trajectory."""
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

        # Build the system prompt for OpenAI (fixed for the episode)
        tools_str = convert_tools_dict_to_str(state.tools)
        openai_system_prompt = CLAUDE_SYSTEM_PROMPT_TEMPLATE_HANDOFF.format(
            system_prompt=state.system_prompt,
            tools=tools_str,
        )

        # Build the policy system prompt
        policy_system_prompt = POLICY_CHAT_SYSTEM_PROMPT_TEMPLATE.format(
            system_prompt=state.system_prompt,
            instruction_prompt=state.new_messages[-1]["content"],
            tools=tools_str,
        )

        # OpenAI maintains conversation state via explicit message list
        openai_messages: list[dict[str, Any]] = []

        while not done:
            # 1. Parse state
            all_messages, new_messages = self._get_messages_from_state(
                state=state, include_system_prompt=False,
            )

            # 2. Policy LLM generates a chat message (tip / reflection)
            if not self.no_cria_prompt:
                if len(all_messages) == 1:
                    _all_messages = []
                else:
                    _all_messages = deepcopy(all_messages)
                    _all_messages = [
                        {"role": "user" if msg["role"] == "assistant" else msg["role"], "content": msg["content"]}
                        for msg in _all_messages
                    ]
                policy_messages = [
                    {"role": "system", "content": policy_system_prompt},
                    *_all_messages,
                    {"role": "user", "content": POLICY_HANDOFF_INSTRUCTION},
                ]
                try:
                    policy_text, state_ids, state_action_ids, action_logprobs = (
                        await self._generate_policy_chat(
                            llm=llm,
                            hf_tokenizer=hf_tokenizer,
                            state=state,
                            all_messages=policy_messages,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            tinker_timeout=self.tinker_timeout,
                        )
                    )
                except Exception as e:
                    logger.error("Policy chat generation failed: %s", e)
                    done = True
                    truncated = True
                    break

                if self.verbose and generation_id == 0 and sample_id == 0:
                    rich_print(
                        f"[bold cyan]Policy Generation:[/bold cyan]\n{policy_text}"
                    )

                policy_answer = (
                    policy_text.split("<final_response>")[1].split("</final_response>")[0]
                ).strip()
                policy_messages_for_step = [{"role": "user", "content": policy_text}]
            else:
                policy_answer = None
                policy_messages_for_step = [{"role": "user", "content": ""}]
                state_action_ids = []
                state_ids = []
                action_logprobs = []

            # 3. Query OpenAI for action
            # Add new environment messages to the OpenAI conversation
            for msg in new_messages:
                openai_messages.append(msg)

                if msg.get("type", None) == "function_call_output":
                    breakpoint()

            try:
                openai_action_messages = await self._get_openai_action(
                    system_prompt=openai_system_prompt,
                    messages=openai_messages,
                    tools=state.tools,
                    policy_text=policy_answer,
                    sample_id=sample_id,
                )
            except RuntimeError:
                logger.warning("OpenAI action failed after retries, ending episode")
                done = True
                truncated = True
                break

            if self.verbose and generation_id == 0 and sample_id == 0:
                rich_print(
                    f"[bold magenta]OpenAI Action:[/bold magenta]"
                    f"\n{openai_action_messages[0]['content']}"
                )

            # Add assistant response to OpenAI conversation history
            openai_messages.extend(openai_action_messages)

            parsed_actions = get_actions(openai_action_messages)

            # 4. Step through environment
            env_step_result: EnvironmentStepResult = await env.step_async(
                parsed_actions=parsed_actions,
                model_response=openai_action_messages,
                current_state=state,
                current_messages=all_messages,
            )
            next_state = env_step_result.state
            reward = env_step_result.reward
            done = env_step_result.done
            truncated = env_step_result.truncated

            # 5. Save EpisodeStep — training on policy LLM's chat tokens
            next_obs = [
                {
                    "role": msg["role"],
                    "content": msg["output"] if msg.get("output", None) else msg["content"],
                }
                for msg in next_state.new_messages
            ]
            episode_steps.append(
                EpisodeStep(
                    state=policy_messages_for_step,
                    action=policy_messages_for_step[0],
                    next_obs=openai_action_messages + next_obs,
                    tools=[],
                    state_action_tokens=state_action_ids,
                    state_len=len(state_ids),
                    old_logprobs=action_logprobs,
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
                    is_complete=True,
                )
            )

            # Add environment responses to OpenAI conversation history
            for msg in next_state.new_messages:
                if msg.get("type", "") == "function_call_output":
                    openai_messages.append({k: v for k, v in msg.items() if k != "role"})
                else:
                    openai_messages.append(msg)

            if self.verbose:
                _header_text = (
                    f"(Method: ChatOpenAI) "
                    f"{split.title()} Split, Try {try_step}, Batch {batch_id},"
                    f" Sample {sample_id}, Generation {generation_id},"
                    f" Timestep {state.timestep}"
                )
                display_state_action_next_obs(
                    generator=self,
                    generation_id=generation_id,
                    state_messages=all_messages,
                    action_messages=policy_messages_for_step,
                    next_obs_messages=openai_action_messages + next_obs,
                    tools=[],
                    hf_tokenizer=hf_tokenizer,
                    header_text=_header_text,
                    group_rewards=[reward],
                    state=state,
                )

            state = next_state
            final_reward = reward
            done = done or truncated

        return Trajectory(
            episode_steps=episode_steps,
            try_step=try_step,
            discount_factor=self.discount_factor,
            final_reward=final_reward,
        )
