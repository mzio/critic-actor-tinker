"""
Chat Claude Generator

Flow:
1. Policy LLM generates a chat message (tip / reflection) given the current state
2. Chat message is injected into Claude's system prompt as context
3. Claude produces a single tool-call action informed by the chat message
4. Environment steps with Claude's action → reward
5. Training signal: policy LLM's generated chat tokens, reward from env.step
"""

import asyncio
import json
import logging
from copy import deepcopy
from typing import Any

from rich import print as rich_print

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
from tinker import SamplingParams
from tinker.types import ModelInput, SampleResponse
from transformers import PreTrainedTokenizerBase

from critic_actor.environments import Environment, EnvironmentState, EnvironmentStepResult
from critic_actor.llm_handlers import TinkerCompleter
from critic_actor.llm_handlers.utils import get_actions
from critic_actor.replay_buffer.types import (
    Trajectory,
    EpisodeStep,
)
from critic_actor.utils.display import display_state_action_next_obs

from ..base import TinkerGenerator
from ..cria_claude.utils import (
    convert_tools_dict_to_str,
    display_actions,
    get_actions_from_response,
    get_prompt_from_messages,
    sample_client_response,
)
from ..cria_claude.types import ClaudeAgentResponse

# from .prompts import get_claude_system_prompt_template, get_policy_chat_instruction, get_claude_chat_prefix
from .prompts import (
    get_claude_system_prompt_template,
    POLICY_CHAT_SYSTEM_PROMPT_TEMPLATE,
    CLAUDE_SYSTEM_PROMPT_TEMPLATE_CHAT,
    CLAUDE_SYSTEM_PROMPT_TEMPLATE_HANDOFF,
)

logger = logging.getLogger(__name__)



POLICY_HANDOFF_INSTRUCTION = """

Now, you are handing off this task to a colleague who will execute the next step.
Write a short, structured handoff note so they can act immediately.

Your note may contain any of the following:
1. **Situation** — one or two sentences summarizing where things stand right now
   (what the task is, what has already been done, what the key outcomes from the last response were).
2. **Key Findings** — bullet the most important facts, numbers, or observations
   from the conversation so far that are relevant to the next step.
3. **Tips** — any tips for the next step. Do not include any explicit action instructions.
   Be specific enough that your colleague can act without re-reading the full conversation.
4. **Reframing** - reframe the system prompt to be more helpful, or synthesize any past tries and their outcomes into actionable advice.

Think and reason about how to provide the best handoff note. Remember it should be concise (<100 words). 

Then include your final handoff note in the following format:
<final_response>
{final-response}
</final_response>
""".strip()


# Per-turn prefix injected before the conversation prompt
CLAUDE_HANDOFF_PREFIX_CHAT = """
[Handoff Note]
{policy_text}
""".strip()
    

def get_policy_chat_messages(message: dict[str, str]) -> list[dict[str, str]]:
    """
    Return user message for policy chat LLM
    """
    messages = []
    if message.get("role", "user") == "user":
        content = message.get("content", "")
        content += f"\n\n{POLICY_HANDOFF_INSTRUCTION}"
    else:
        assert message.get("role", None) == "tool", "Expected tool response message if not user"
        messages.append(message)
        content = POLICY_HANDOFF_INSTRUCTION
    messages.append({"role": "user", "content": content})
    return messages


class ChatClaudeResponseSingleAction:
    """Schema for Claude's single-action structured output."""

    @staticmethod
    def model_json_schema() -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "tool_call": {"type": "string"},
            },
            "required": ["reasoning", "tool_call"],
        }


class ChatClaudeGenerator(TinkerGenerator):
    """
    Chat Claude Generator: policy LLM produces a chat message (tip / reflection),
    which is injected into Claude's prompt. Claude then acts, and the reward
    trains the policy on its chat generation.
    """

    def __init__(
        self,
        prompt_template_name: str = "chat",
        # Claude Agent SDK options
        model: str = "claude-haiku-4-5",
        permission_mode: str = "default",
        effort: str = "low",
        max_agent_turns: int | None = 1,
        # Tinker options
        tinker_timeout: int = 300,
        no_cria_prompt: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.prompt_template_name = prompt_template_name

        # Claude Agent SDK
        self.claude_option_kwargs = {
            "model": model,
            "permission_mode": permission_mode,
            "effort": effort,
            "max_turns": max_agent_turns,
            # "output_format": {
            #     "type": "json_schema",
            #     "schema": ChatClaudeResponseSingleAction.model_json_schema(),
            # },
            "allowed_tools": [],
            "tools": [],
        }
        self.claude_system_prompt_template = get_claude_system_prompt_template(
            prompt_template_name
        )
        # Cumulative cost tracking
        self._cumulative_cost_usd: float = 0.0
        self._batch_cost_usd: float = 0.0

        # Tinker
        self.tinker_timeout = tinker_timeout
        self.no_cria_prompt = no_cria_prompt


    def get_cost_metrics(self) -> dict[str, float]:
        """Return cost metrics and reset batch cost."""
        metrics = {
            "cost/batch_usd": self._batch_cost_usd,
            "cost/cumulative_usd": self._cumulative_cost_usd,
        }
        self._batch_cost_usd = 0.0
        return metrics

    def _track_cost(self, cost: float | None) -> None:
        if cost and cost > 0:
            self._cumulative_cost_usd += cost
            self._batch_cost_usd += cost

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
    # Policy LLM generation
    # ------------------------------------------------------------------
    async def _generate_policy_chat(
        self,
        llm: TinkerCompleter,
        hf_tokenizer: PreTrainedTokenizerBase,
        state: EnvironmentState,
        all_messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        tinker_timeout: int | None,
        continue_prompt: str | None = "Okay, let me think about"
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

        # Tokenize state (with generation prompt for the assistant turn)
        all_messages = deepcopy(all_messages)
        if continue_prompt:
            all_messages.append({"role": "assistant", "content": continue_prompt})
        state_ids: list[int] = hf_tokenizer.apply_chat_template(
            conversation=all_messages,
            # tools=state.tools,
            add_generation_prompt=continue_prompt is None,
            continue_final_message=continue_prompt is not None,
            enable_thinking=self.enable_thinking,
            tokenize=True,
            return_dict=False,
        )
        state_tinker_input = ModelInput.from_ints(state_ids)

        # Generate chat message from policy LLM
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
            assert sampled_logprobs is not None, "No logprobs found in policy model generation"

            # Decode the chat message
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

        # Build state_action_ids for training (state + policy chat generation)
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

        # Recompute logprobs over the full state + action sequence
        all_logprobs = await llm.compute_logprobs_async(state_action_tinker_input)
        action_token_len = len(state_action_ids) - len(state_ids)
        action_logprobs: list[float] = list(all_logprobs[-action_token_len:])

        return model_text, state_ids, state_action_ids, action_logprobs

    # ------------------------------------------------------------------
    # Claude action from policy chat
    # ------------------------------------------------------------------
    async def _get_claude_action(
        self,
        client: ClaudeSDKClient,
        new_messages: list[dict[str, Any]],
        policy_text: str | None,
        sample_id: int,
    ) -> tuple[list[dict[str, str]], ClaudeAgentResponse]:
        """
        Query Claude with the conversation context plus the policy's chat
        message prepended. The client is long-lived across timesteps so
        Claude accumulates conversation history.

        Returns (model_messages, claude_response).
        """
        # Prepend policy chat message to the per-turn prompt
        # chat_prefix = get_claude_chat_prefix(chat_text, name=self.prompt_template_name)
        if policy_text is not None:
            conversation_prompt = get_prompt_from_messages(new_messages)
            chat_suffix = CLAUDE_HANDOFF_PREFIX_CHAT.format(policy_text=policy_text)
            claude_prompt = f"{conversation_prompt}\n\n{chat_suffix}"
        else:
            claude_prompt = get_prompt_from_messages(new_messages)

        num_attempts = 0
        while num_attempts < 10:
            try:
                claude_response = await sample_client_response(client, claude_prompt)
                self._track_cost(claude_response.cost)
                actions = get_actions_from_response(claude_response)

                if self.verbose:  #  and sample_id == 0:
                    rich_print(f"[dim]Claude Prompt:\n{claude_prompt}[/dim]")
                    display_actions(actions, base_color=f"color({(sample_id + 1) % 8 + 8})")

                for action in actions:
                    if action.type == "function_call" and action.name == "StructuredOutput":
                        reasoning = action.arguments.get("reasoning", "")
                        tool_call = action.arguments.get("tool_call", "")
                        if "<tool_call>" not in tool_call or "</tool_call>" not in tool_call:
                            raise ValueError(
                                f"Missing <tool_call> tags in: {tool_call}"
                            )
                        content = f"{reasoning}\n\n{tool_call}"
                        return [{"role": "assistant", "content": content}], claude_response

                    elif action.type == "message":
                        content = action.text
                        return [{"role": "assistant", "content": content}], claude_response

                raise ValueError("No StructuredOutput action found in Claude response")

            except Exception as e:
                logger.error("Claude action parse error (%s): %s", e.__class__.__name__, e)
                claude_prompt = f"Response not parsed correctly, please try again.\n\n{e}"
                num_attempts += 1

        # Exhausted retries
        raise RuntimeError("Failed to get valid Claude action after 10 attempts")

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

        # Set up Claude once — system prompt is fixed for the entire episode.
        # The policy's per-turn chat message goes into the user prompt instead.
        tools_str = convert_tools_dict_to_str(state.tools)
        claude_system_prompt = self.claude_system_prompt_template.format(
            system_prompt=state.system_prompt,
            tools=tools_str,
        )
        claude_option_kwargs = deepcopy(self.claude_option_kwargs)
        claude_option_kwargs["system_prompt"] = claude_system_prompt
        claude_options = ClaudeAgentOptions(**claude_option_kwargs)

        policy_system_prompt = POLICY_CHAT_SYSTEM_PROMPT_TEMPLATE.format(
            system_prompt=state.system_prompt,
            instruction_prompt=state.new_messages[-1]["content"],
            tools=tools_str,
        )

        async with ClaudeSDKClient(options=claude_options) as client:
            while not done:
                # 1. Parse state
                all_messages, new_messages = self._get_messages_from_state(
                    state=state, include_system_prompt=False,
                )  # Can extend this with retrieval from past actions

                # 2. Policy LLM generates a chat message (tip / reflection)
                if not self.no_cria_prompt:
                    # Process all_messages for our policy LLM
                    assert all_messages[-1] == new_messages[0], (
                        "Expected last message in all_messages to match new_messages[0]"
                    )
                    # _all_messages = deepcopy(all_messages[:-1]) if len(all_messages) > 1 else []
                    if len(all_messages) == 1:
                        _all_messages = []  # First prompt, instruction is in policy_system_prompt
                    else:
                        # Swap user and assistant roles for policy LLM
                        _all_messages = deepcopy(all_messages)
                        _all_messages = [
                            {"role": "user" if msg["role"] == "assistant" else msg["role"], "content": msg["content"]}
                            for msg in _all_messages
                        ]
                    policy_messages = [
                        {"role": "system", "content": policy_system_prompt},
                        # *_all_messages,  # all except the newest message
                        # *get_policy_chat_messages(all_messages[-1]),  # formatted newest message
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
                        assert "<final_response>" in policy_text and "</final_response>" in policy_text, (
                            "Expected <final_response> tags in policy model generation"
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

                    # 3. Query Claude — policy chat is prepended to the per-turn prompt
                    policy_answer = (
                        policy_text.split("<final_response>")[1].split("</final_response>")[0]
                    ).strip()
                    policy_messages = [{"role": "user", "content": policy_text}]  # Save full text for replay buffer
                else:
                    policy_answer = None
                    policy_messages = [{"role": "user", "content": ""}]  # dummy holder (we only do this during eval)
                    state_action_ids = []
                    state_ids = []
                    action_logprobs = []

                try:
                    claude_messages, _ = await self._get_claude_action(
                        client=client,
                        new_messages=deepcopy(new_messages),
                        policy_text=policy_answer,
                        sample_id=sample_id,
                    )
                except RuntimeError:
                    logger.warning("Claude action failed after retries, ending episode")
                    done = True
                    truncated = True
                    break

                if self.verbose and generation_id == 0 and sample_id == 0:
                    rich_print(
                        f"[bold magenta]Claude Action:[/bold magenta]"
                        f"\n{claude_messages[0]['content']}"
                    )

                parsed_actions = get_actions(claude_messages)

                # 4. Step through environment with Claude's action
                env_step_result: EnvironmentStepResult = await env.step_async(
                    parsed_actions=parsed_actions,
                    model_response=claude_messages,
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
                        # state=all_messages,
                        # action=model_messages[0],
                        # next_obs=next_obs,
                        state=policy_messages,
                        action=policy_messages[0],
                        next_obs=claude_messages + next_obs,
                        tools=[],  # state.tools,
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

                # # Feed Claude's chosen action back as a user message for next turn
                # try:
                #     _tool_call = (
                #         model_messages[0]["content"]
                #         .split("<tool_call>")[1]
                #         .split("</tool_call>")[0]
                #     )
                # except IndexError:
                #     logger.warning("Could not parse tool_call from Claude response")
                #     done = True
                #     truncated = True
                #     break
                # user_reply = f"Thanks, I chose:\n<tool_call>{_tool_call}</tool_call>"
                # next_state.new_messages.insert(0, {"role": "user", "content": user_reply})

                if self.verbose:
                    _header_text = (
                        f"(Method: ChatClaude) "
                        f"{split.title()} Split, Try {try_step}, Batch {batch_id},"
                        f" Sample {sample_id}, Generation {generation_id},"
                        f" Timestep {state.timestep}"
                    )
                    display_state_action_next_obs(
                        generator=self,
                        generation_id=generation_id,
                        state_messages=all_messages,
                        action_messages=policy_messages,
                        next_obs_messages=claude_messages + next_obs,
                        tools=[],  # state.tools,
                        hf_tokenizer=hf_tokenizer,
                        header_text=_header_text,
                        group_rewards=[reward],  # Should we group rewards?
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
