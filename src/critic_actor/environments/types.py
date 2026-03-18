"""
Environment objects and types
"""

from typing import Any

from pydantic import BaseModel

from datasets import Dataset
from datasets.iterable_dataset import IterableColumn, IterableDataset

# Type suggestions
type Message = dict[str, Any]
type Tool = dict[str, Any]
type ModelResponse = Any
type PriorMessage = Any
# HuggingFace Datasets
type HuggingFaceDatasetOrColumn = Dataset | IterableDataset | IterableColumn


class EnvironmentState(BaseModel):
    """
    State of the environment after a step
    """

    system_prompt: str
    new_messages: list[Message]
    model_response: ModelResponse | None
    prior_messages: list[PriorMessage]
    tools: list[Tool]
    # Other metadata / info
    sample_id: int = 0
    generation_id: int = 0
    batch_id: int = 0
    timestep: int = 0
    try_step: int = 0
    metadata: dict[str, Any] | None = None
    # Number of past observations to show
    first_obs_to_show: int | None = None
    last_obs_to_show: int | None = None

    # Other metadata helpful for ICL or retrieval
    task_prompt: str | None = None  # e.g., the content in the initial user message
    default_context: list[Message] | None = None  # e.g., list corresponding to few-shot examples


class EnvironmentStateWithAnswer(EnvironmentState):
    """
    State of the environment after a step for QA tasks
    """

    question: str | None
    answer: str | None
    grading_rubric: list[dict[str, Any]] | None = None


class EnvironmentStepResult(BaseModel):
    """
    Result of a step in the environment
    - Note: we may not use, as most Gym conventions just return these as a tuple
    """

    state: EnvironmentState
    reward: float
    done: bool
    truncated: bool
    info: dict[str, Any] | None = None


# Alternative to above:
# We can use this type suggestion for environment.step() return value
type StepResult = tuple[
    EnvironmentState | EnvironmentStateWithAnswer,
    float,
    bool,
    bool,
    dict[str, Any],
]
