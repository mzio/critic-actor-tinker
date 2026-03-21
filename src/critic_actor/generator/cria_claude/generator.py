"""
Critic-Actor Generator with Claude (via Claude Agent SDK)
"""

import asyncio
import json
import logging
from copy import deepcopy
from typing import Any

import numpy as np
from rich import print as rich_print

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
from tinker.types import ModelInput
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

from .utils import (
    convert_tools_dict_to_str,
    display_actions,
    get_actions_from_response,
    get_prompt_from_messages,
    sample_client_response,
)
from .prompts import get_system_prompt_template
from .types import ClaudeResponseForCriticActor

logger = logging.getLogger(__name__)
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


CRITIC_ACTOR_PROMPT_TEMPLATE_MULTIPLE_CHOICE = """
Let me consider the following options:

{multiple_choice_text}

Final choice: {final_choice}
"""

class CriticActorClaudeGenerator(TinkerGenerator):
    """
    Critic-Actor Generator with Claude (via Claude Agent SDK)
    """
    def __init__(
        self,
        num_actions: int = 2,
        prompt_template_name: str = "human_tool",
        use_claude_tools: bool = False,  # True not supported yet
        # Claude Agent SDK options
        model: str = "claude-haiku-4-5",
        permission_mode: str = "default",
        effort: str = "low",
        max_agent_turns: int | None = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        
        self.num_actions = num_actions
        self.prompt_template_name = prompt_template_name
        self.use_claude_tools = use_claude_tools
        
        self.claude_option_kwargs = {
            "model": model,
            "permission_mode": permission_mode,
            "effort": effort,
            "max_turns": max_agent_turns,
            "output_format": {
                "type": "json_schema",
                "schema": ClaudeResponseForCriticActor.model_json_schema(),
            },
        }
        if not self.use_claude_tools:
            self.claude_option_kwargs.update({
                "allowed_tools": [],
                "tools": [],
            })
        self.system_prompt_template = get_system_prompt_template(prompt_template_name)

        # Cumulative cost tracking
        self._cumulative_cost_usd: float = 0.0
        self._batch_cost_usd: float = 0.0  # reset per batch for per-step logging

    def get_cost_metrics(self) -> dict[str, float]:
        """Return cost metrics and reset batch cost. Cumulative keeps accumulating."""
        metrics = {
            "cost/batch_usd": self._batch_cost_usd,
            "cost/cumulative_usd": self._cumulative_cost_usd,
        }
        self._batch_cost_usd = 0.0
        return metrics

    def _track_cost(self, cost: float | None) -> None:
        """Accumulate cost from a single Claude response."""
        if cost and cost > 0:
            self._cumulative_cost_usd += cost
            self._batch_cost_usd += cost

    def _get_messages_from_state(
        self,
        state: EnvironmentState,
        include_system_prompt: bool = False,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Get messages from the environment state, in the form of
        [{"role": <role>, "content": <content>}, ...]

        Returns both (messages, new_messages)
        """
        # First handle new_messages
        new_messages = [
            {"role": msg["role"], "content": msg["output"]}
            if msg.get("type", "") == "function_call_output"
            else msg
            for msg in state.new_messages
        ]
        # Then handle all_messages
        model_response: list[dict[str, str]] = state.model_response or []
        all_messages = (
            (state.prior_messages or [])
            + model_response
            + new_messages  # state.new_messages
        )
        # Remove system prompt if present and undesired
        if all_messages[0].get("role", "") == "system" and not include_system_prompt:
            all_messages = all_messages[1:]
        # Return all_messages and new_messages lists
        return all_messages, new_messages

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

        # Make Claude a harness for Critic-Actor
        system_prompt: str = self.system_prompt_template.format(
            num_actions=self.num_actions,
            system_prompt=state.system_prompt,
            tools=convert_tools_dict_to_str(state.tools),
        )
        claude_option_kwargs = deepcopy(self.claude_option_kwargs)
        claude_option_kwargs.update({
            "system_prompt": system_prompt,
        })
        claude_options = ClaudeAgentOptions(**claude_option_kwargs)

        async with ClaudeSDKClient(options=claude_options) as client:
            # Generate model responses and step through the environment
            while not done:
                # Parse state messages
                state_messages_tuple = self._get_messages_from_state(
                    state=state,
                    include_system_prompt=False,
                )
                all_messages = state_messages_tuple[0]
                new_messages = state_messages_tuple[1]
                
                cria_actions = []
                incorrect_num_actions = False
                incorrect_tool_format = False
                num_attempts = 0
                while len(cria_actions) == 0:
                    try:
                        claude_prompt = get_prompt_from_messages(new_messages)
                        claude_response = await sample_client_response(client, claude_prompt)
                        self._track_cost(claude_response.cost)
                        actions = get_actions_from_response(claude_response)
                        if self.verbose and generation_id == 0 and sample_id == 0:
                            rich_print(f"[bold]Claude Prompt:\n{claude_prompt}[/bold]")
                            display_actions(actions, base_color=f"color({(sample_id + 1) % 8 + 8})")
                        for action in actions:
                            if action.type == "function_call" and action.name == "StructuredOutput":
                                # Check number of actions matches
                                _messages = action.arguments["messages"]
                                if len(_messages) != self.num_actions:
                                    if isinstance(_messages, str):
                                        try:
                                            _messages = json.loads(_messages)
                                            if len(_messages) != self.num_actions:
                                                raise ValueError(
                                                    f"Expected {self.num_actions} messages, got {len(_messages)}"
                                                )
                                            incorrect_num_actions = False
                                        except Exception as e:
                                            incorrect_num_actions = True
                                            raise e
                                    else:
                                        incorrect_num_actions = True
                                        raise ValueError(
                                            f"Expected {self.num_actions} messages, got {len(action.arguments['messages'])}"
                                        )
                                for _msg in _messages:
                                    if "<tool_call>" not in _msg["tool_call"] or "</tool_call>" not in _msg["tool_call"]:
                                        incorrect_tool_format = True
                                        raise ValueError(
                                            f"Expected tool call format, got {_msg['tool_call']} (missing <tool_call> or </tool_call>)"
                                        )
                                cria_actions.extend(_messages)  # Should be structured output
                                break
                    except Exception as e:
                        logger.error(
                            "Error in sample_client_response (%s): %s", e.__class__.__name__, e
                        )
                        # if incorrect_num_actions:
                        #     content = f"Sorry, incorrect number of actions (needed {self.num_actions})! Please try again."
                        #     content += f"\n\n{e.__class__.__name__}: {e}"
                        # elif incorrect_tool_format:
                        #     content = (
                        #         "Sorry, incorrect tool call format! Need proper <tool_call> and </tool_call> tags."
                        #         " Please try again."
                        #     )
                        # else:
                        #     content = "Sorry, response not parsed correctly! Please try again."
                        content = f"Sorry, response not parsed correctly! Please try again.\n\n{e.__class__.__name__}: {e}"
                        new_messages.append({
                            "role": "user", "content": content
                        })
                        num_attempts += 1
                        if num_attempts >= 10:
                            done = True
                            reward = -1.0
                            truncated = True
                            break
                        
                # If max retry attempts exhausted, just end the episode with existing steps
                if done and len(cria_actions) == 0:
                    logger.warning("Max retry attempts reached, ending episode early")
                    break

                # First format each potential action into standard assistant messages chat
                _assistant_template = "{reasoning}\n\n{tool_call}"
                assistant_messages: list[list[dict[str, str]]] = [
                    [{"role": "assistant", "content": _assistant_template.format(**cria_action)}]
                    for cria_action in cria_actions
                ]
                # Then format into Critic-Actor samples
                cria_msg_parts = []
                for _idx, _msgs in enumerate(assistant_messages):
                    cria_msg_parts.append(f"{ALPHABET[_idx]}) {_msgs[0]['content']}")
                mc_action_text = "\n\n".join(cria_msg_parts)

                cria_prompt_template = CRITIC_ACTOR_PROMPT_TEMPLATE_MULTIPLE_CHOICE
                # Get input ids without final choice
                cria_prefix_chat = deepcopy(all_messages) + [{
                    "role": "assistant",
                    "content": cria_prompt_template.format(
                        multiple_choice_text=mc_action_text,
                        final_choice=""
                    ),
                }]
                # Tokenize
                shared_tokenizer_kwargs = {
                    "tools": state.tools,
                    "add_generation_prompt": False,
                    "continue_final_message": True,  # no <eos>
                    "enable_thinking": False,
                    "tokenize": True,
                    "return_dict": False,
                }
                cria_prefix_ids: list[int] = hf_tokenizer.apply_chat_template(
                    conversation=cria_prefix_chat,
                    **shared_tokenizer_kwargs,
                )
                cria_prefix_len = len(cria_prefix_ids)
                # Compare all actions
                cria_action_chats = [deepcopy(all_messages) for _ in range(self.num_actions)]
                for _idx in range(self.num_actions):
                    choice = ALPHABET[_idx]
                    cria_prompt = cria_prompt_template.format(
                        multiple_choice_text=mc_action_text,
                        final_choice=choice,
                    )
                    cria_action_chats[_idx].append({"role": "assistant", "content": cria_prompt})
                cria_completion_ids = [
                    hf_tokenizer.apply_chat_template(
                        conversation=cria_action_chat,
                        **shared_tokenizer_kwargs,
                    )
                    for cria_action_chat in cria_action_chats
                ]
                if self.verbose:
                    for _completion_idx, _completion_ids in enumerate(cria_completion_ids):
                        _color = f"color({(_completion_idx + 1) % 8 + 8})"
                        _completion_text = hf_tokenizer.decode(_completion_ids)
                        rich_print(
                            f"[{_color}] Critic-Actor Completion {_completion_idx}:"
                            f"\n{_completion_text} [/{_color}]"
                        )
                tinker_cria_completion_ids = [
                    ModelInput.from_ints(_input_ids) for _input_ids in cria_completion_ids
                ]
                # Compute logprobs for all critic-actor completions
                # Wrap each in try/except to handle context window overflow gracefully
                async def _safe_logprobs(idx: int) -> list[float] | None:
                    try:
                        return await asyncio.wait_for(
                            llm.compute_logprobs_async(tinker_cria_completion_ids[idx]),
                            timeout=120,
                        )
                    except asyncio.TimeoutError:
                        logger.warning("Action %d compute_logprobs timed out after 120s", idx)
                        return None
                    except Exception as e:
                        if "context window" in str(e) or "max_tokens" in str(e):
                            logger.warning("Action %d exceeds context window, skipping: %s", idx, e)
                            return None
                        raise

                all_complete_logprobs = await asyncio.gather(*[
                    _safe_logprobs(_idx) for _idx in range(self.num_actions)
                ])
                # Select action based on highest logprob (-inf for failed candidates)
                all_cls_logprobs = [
                    np.array(logprobs)[cria_prefix_len:].mean().item()
                    if logprobs is not None else -float("inf")
                    for logprobs in all_complete_logprobs
                ]
                # If ALL candidates failed, end the episode
                if all(lp == -float("inf") for lp in all_cls_logprobs):
                    logger.warning("All action candidates exceed context window, ending episode")
                    done = True
                    truncated = True
                    break
                best_choice_idx = np.argmax(all_cls_logprobs)
                try:
                    model_messages = assistant_messages[best_choice_idx]
                except Exception as e:
                    print(f"Error in assistant_messages[best_choice_idx]: {e.__class__.__name__}: {e}")
                    breakpoint()
                parsed_actions = get_actions(model_messages)

                # Step thru environment
                env_step_result: EnvironmentStepResult = await env.step_async(
                    parsed_actions=parsed_actions,
                    model_response=model_messages,
                    current_state=state,
                    current_messages=all_messages,
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
                other_completion_ids = [
                    cria_completion_ids[_idx] for _idx in range(self.num_actions) if _idx != best_choice_idx
                ]
                other_completion_ids = other_completion_ids if len(other_completion_ids) > 0 else None
                episode_steps.append(
                    EpisodeStep(
                        # Note we don't use [state, action, next_obs, tools] for training
                        state=all_messages,  # list[dict[str, str]]
                        action=model_messages[0],  # dict[str, str]
                        # action=cria_action_chats[best_choice_idx][-1],
                        next_obs=next_obs,  # list[dict[str, str]]
                        tools=state.tools,
                        state_action_tokens=cria_completion_ids[best_choice_idx],
                        other_state_action_tokens=other_completion_ids,
                        state_len=cria_prefix_len,
                        old_logprobs=[all_cls_logprobs[best_choice_idx]],
                        # Should also figure out how to incorporate the other options for training
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
                # Add chosen message as a user message
                try:
                    _tool_call = model_messages[0]["content"].split("<tool_call>")[1].split("</tool_call>")[0]
                except IndexError:
                    breakpoint()
                user_reply = f"Thanks, I chose:\n<tool_call>{_tool_call}</tool_call>"
                next_state.new_messages.insert(0, {"role": "user", "content": user_reply})

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
                        state_messages=all_messages,
                        action_messages=model_messages,
                        # action_messages=[cria_action_chats[best_choice_idx][-1]],
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
