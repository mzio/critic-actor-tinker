"""
GDPval RLIC Environment
=======================

Wraps OpenAI's GDPval benchmark as an RLIC Environment. Each episode:
1. ``reset()``: creates workdir, copies reference files, builds system prompt.
2. ``step(execute_python)``: executes Python in a persistent namespace.
3. ``step(submit)`` or truncation: grades deliverables via LLM rubric grader.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from copy import copy
from pathlib import Path
from typing import Any

import numpy as np

from ...llm_handlers import ActionFromLLM
from ..base import Environment
from ..types import EnvironmentState, EnvironmentStepResult
from .data import GDPvalTask, load_gdpval_dataset, split_tasks
from .executor import CodeExecutor
from .grader import GDPvalGrader
from .prompts import TOOLS, build_system_prompt
from .utils import collect_deliverables, extract_action, format_exec_result

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State / result types
# ---------------------------------------------------------------------------


class GDPvalState(EnvironmentState):
    """RLIC state with GDPval-specific metadata."""

    task_id: str
    occupation: str = ""
    sector: str = ""


class GDPvalStepResult(EnvironmentStepResult):
    """Step result with typed state."""

    state: GDPvalState


# ---------------------------------------------------------------------------
# Main environment
# ---------------------------------------------------------------------------


class GDPvalEnv(Environment):
    """RLIC environment for GDPval professional tasks.

    Args:
        cache_dir: Directory for caching HF dataset and reference files.
        num_train_samples: Number of tasks for the training split.
        num_eval_samples: Number of tasks for the eval split.
        max_turns: Maximum agent turns per episode.
        code_timeout_sec: Timeout for each code execution.
        grader_model_config: Config dict for the grader LLM.
        grader_batch_criteria: If True, grade all criteria in one LLM call.
        **kwargs: Forwarded to :class:`~critic_actor.environments.base.Environment`.
    """

    def __init__(
        self,
        cache_dir: str,
        num_train_samples: int = 180,
        num_eval_samples: int = 40,
        max_turns: int = 30,
        code_timeout_sec: int = 60,
        grader_model_config: dict[str, Any] | None = None,
        grader_batch_criteria: bool = True,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("split", "train")
        kwargs.setdefault("eval_splits", ["eval"])
        kwargs.setdefault("max_turns", max_turns)

        super().__init__(**kwargs)

        self.cache_dir = Path(cache_dir)
        self.num_train_samples = num_train_samples
        self.num_eval_samples = num_eval_samples
        self.max_turns = max_turns
        self.code_timeout_sec = code_timeout_sec
        self.grader_model_config = grader_model_config
        self.grader_batch_criteria = grader_batch_criteria

        # Per-episode state
        self._executor: CodeExecutor | None = None
        self._workdir: Path | None = None
        self._current_task: GDPvalTask | None = None
        self._reference_filenames: set[str] = set()

        # Lazy-init grader (needs LLM import)
        self._grader: GDPvalGrader | None = None

        # Load and split tasks
        self.datasets: dict[str, list[GDPvalTask]] = self._load_datasets()

    def _load_datasets(self) -> dict[str, list[GDPvalTask]]:
        """Load GDPval tasks and split into train/eval."""
        tasks = load_gdpval_dataset(self.cache_dir)
        return split_tasks(
            tasks,
            num_train=self.num_train_samples,
            num_eval=self.num_eval_samples,
            seed=self.data_seed,
        )

    def _get_grader(self) -> GDPvalGrader:
        """Lazy-initialize the grader."""
        if self._grader is None:
            self._grader = GDPvalGrader(
                grader_model_config=self.grader_model_config,
                batch_criteria=self.grader_batch_criteria,
            )
        return self._grader

    # ------------------------------------------------------------------
    # Environment protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.datasets[self.split])

    def adjust_sample_id(self, sample_idx: int) -> int:
        n = len(self.datasets[self.split])
        if n == 0:
            raise RuntimeError(
                f"gdpval: no tasks loaded for split='{self.split}'. "
                f"Check cache_dir={self.cache_dir}"
            )
        return sample_idx % n

    def shuffle(self, seed: int | None = None) -> None:
        rng = np.random.default_rng(seed if seed is not None else self.data_seed)
        tasks = self.datasets[self.split]
        indices = rng.permutation(len(tasks)).tolist()
        self.datasets[self.split] = [tasks[i] for i in indices]

    # ------------------------------------------------------------------
    # Episode lifecycle helpers
    # ------------------------------------------------------------------

    def _cleanup_episode(self) -> None:
        """Clean up workdir and executor from previous episode."""
        self._executor = None
        if self._workdir is not None and self._workdir.exists():
            try:
                shutil.rmtree(self._workdir)
            except Exception as e:
                logger.warning("gdpval: failed to clean up workdir: %s", e)
        self._workdir = None
        self._current_task = None
        self._reference_filenames = set()

    def _setup_episode(self, task: GDPvalTask) -> None:
        """Set up workdir, copy reference files, init executor."""
        self._current_task = task

        # Create temp workdir
        self._workdir = Path(tempfile.mkdtemp(prefix=f"gdpval_{task.task_id}_"))

        # Copy reference files (read-only to prevent model overwriting them)
        self._reference_filenames = set()
        for ref_path in task.reference_files:
            dest = self._workdir / ref_path.name
            shutil.copy2(ref_path, dest)
            dest.chmod(0o444)
            self._reference_filenames.add(ref_path.name)

        # Init executor
        self._executor = CodeExecutor(
            workdir=self._workdir,
            timeout_sec=self.code_timeout_sec,
        )

    def _run_grader(self) -> tuple[float, dict[str, Any]]:
        """Grade deliverables for the current episode."""
        task = self._current_task
        if task is None or self._workdir is None:
            return 0.0, {}

        deliverables = collect_deliverables(self._workdir, self._reference_filenames)
        grader = self._get_grader()

        try:
            reward, info = grader.grade(
                task_prompt=task.prompt,
                rubric_json=task.rubric_json,
                deliverable_paths=deliverables,
                total_points=task.total_points,
            )
        except Exception as e:
            logger.warning("gdpval: grader failed: %s", e)
            reward = 0.0
            info = {"error": str(e), "grader_failed": True}

        return reward, info

    # ------------------------------------------------------------------
    # reset()
    # ------------------------------------------------------------------

    def reset(
        self,
        sample_id: int = 0,
        generation_id: int = 0,
        try_step: int = 0,
        batch_id: int = 0,
    ) -> GDPvalState:
        """Start a new episode: clean up, set up workdir, return state."""
        self._cleanup_episode()

        idx = self.adjust_sample_id(sample_id)
        task = self.datasets[self.split][idx]

        # Set up episode
        self._setup_episode(task)

        # Build system prompt
        ref_names = [p.name for p in task.reference_files]
        system_prompt = build_system_prompt(
            task_prompt=task.prompt,
            workdir=str(self._workdir),
            reference_files=ref_names,
        )

        new_messages = [
            {
                "role": "user",
                "content": (
                    "Your task is described in the system prompt. "
                    "Use the `execute_python` tool to write and run Python code, "
                    "and the `submit` tool when you have produced all deliverables."
                ),
            }
        ]

        return GDPvalState(
            system_prompt=system_prompt,
            new_messages=new_messages,
            model_response=None,
            prior_messages=[],
            tools=TOOLS,
            # GDPval-specific
            task_id=task.task_id,
            occupation=task.occupation,
            sector=task.sector,
            # RLIC bookkeeping
            sample_id=sample_id,
            generation_id=generation_id,
            batch_id=batch_id,
            try_step=try_step,
            timestep=0,
            split=self.split,
            task_prompt=task.prompt,
            metadata={"correct": 0, "total": 1},
            first_obs_to_show=2,  # system + initial user message
        )

    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------

    def step(self, **kwargs: Any) -> GDPvalStepResult:
        return self._step_impl(**kwargs)

    def _step_impl(
        self,
        parsed_actions: list[ActionFromLLM],
        model_response: Any,
        current_state: GDPvalState,
        current_messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> GDPvalStepResult:
        """Execute one step: run Python code or submit for grading."""
        if self._executor is None:
            raise RuntimeError("gdpval: step() called before reset().")

        done = False
        truncated = False
        reward = 0.0
        env_messages = []
        grading_info = {}

        code, action_type = extract_action(parsed_actions)

        if action_type == "submit":
            # Grade deliverables
            reward, grading_info = self._run_grader()
            if self.negative_rewards:
                reward = reward * 2.0 - 1.0
            done = True

            earned = grading_info.get("earned_points", 0)
            total = grading_info.get("total_points", 0)
            env_messages.append(
                {
                    "role": "user",
                    "content": f"# GRADING RESULT: {earned}/{total} points (reward={reward:.3f})",
                }
            )

        elif action_type == "execute_python" and code:
            # Execute Python code
            stdout, stderr, return_code = self._executor.execute(code)
            output = format_exec_result(stdout, stderr, return_code)

            # Find the call_id from the parsed action
            call_id = None
            for action in parsed_actions:
                if action.type == "function_call" and action.name == "execute_python":
                    call_id = action.call_id
                    break

            if call_id:
                env_messages.append(
                    {
                        "role": "tool",
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": output,
                    }
                )
            else:
                env_messages.append(
                    {
                        "role": "user",
                        "content": output,
                    }
                )
        else:
            env_messages.append(
                {
                    "role": "user",
                    "content": (
                        "No valid action parsed. Use the `execute_python` tool to run code "
                        "or `submit` when done."
                    ),
                }
            )

        # Update timestep and check truncation
        timestep = current_state.timestep + 1
        if timestep >= self.max_turns and not done:
            truncated = True
            done = True
            # Grade whatever files exist for partial credit
            reward, grading_info = self._run_grader()
            if self.negative_rewards:
                reward = reward * 2.0 - 1.0
            env_messages.append(
                {
                    "role": "user",
                    "content": self.truncation_reply,
                }
            )

        # Clean up when episode is done
        if done:
            self._cleanup_episode()

        # Hide prior observations if configured
        current_messages = self.maybe_hide_observations(
            current_messages or [],
            first_obs_to_show=current_state.first_obs_to_show,
            last_obs_to_show=current_state.last_obs_to_show,
        )

        metadata = copy(current_state.metadata or {})
        metadata["correct"] = int(grading_info.get("earned_points", 0) > 0) if grading_info else int(reward > 0)
        metadata["total"] = 1
        metadata["reward"] = reward
        metadata["done"] = done
        metadata["truncated"] = truncated
        if grading_info:
            metadata["grading_info"] = grading_info
            if grading_info.get("grader_failed") or grading_info.get("error"):
                metadata["grader_failed"] = True

        new_state = GDPvalState(
            system_prompt=current_state.system_prompt,
            new_messages=env_messages,
            model_response=model_response,
            prior_messages=current_messages,
            tools=current_state.tools,
            # GDPval-specific
            task_id=current_state.task_id,
            occupation=current_state.occupation,
            sector=current_state.sector,
            # RLIC bookkeeping
            task_prompt=current_state.task_prompt,
            default_context=current_state.default_context,
            sample_id=current_state.sample_id,
            generation_id=current_state.generation_id,
            batch_id=current_state.batch_id,
            try_step=current_state.try_step,
            timestep=timestep,
            split=self.split,
            metadata=metadata,
            first_obs_to_show=current_state.first_obs_to_show,
            last_obs_to_show=current_state.last_obs_to_show,
        )

        return GDPvalStepResult(
            state=new_state,
            reward=reward,
            done=done,
            truncated=truncated,
            info=metadata,
        )


# ---------------------------------------------------------------------------
# Async wrapper
# ---------------------------------------------------------------------------


class AsyncGDPvalEnv(GDPvalEnv):
    """Async wrapper around GDPvalEnv for compatibility with async generators."""

    async def reset_async(
        self,
        sample_id: int = 0,
        generation_id: int = 0,
        try_step: int = 0,
        batch_id: int = 0,
        **kwargs: Any,
    ) -> GDPvalState:
        return self.reset(
            sample_id=sample_id,
            generation_id=generation_id,
            try_step=try_step,
            batch_id=batch_id,
        )

    async def step_async(self, **kwargs: Any) -> GDPvalStepResult:
        return self.step(**kwargs)
