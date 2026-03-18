"""
GDPval RLIC environment package.

Wraps OpenAI's GDPval benchmark (220 professional tasks across 44 occupations)
as an RLIC Environment. Model iteratively executes Python code via a shared
namespace (like Jupyter) to produce deliverable files, graded by an LLM judge
against detailed rubrics.

Quick start::

    from critic_actor.environments import get_env

    env = get_env("gdpval", cache_dir="/path/to/gdpval_cache")
    state = env.reset(sample_id=0, generation_id=0)

Installation note:
    Install optional deps with: uv sync --group gdpval
"""

from .env import AsyncGDPvalEnv, GDPvalEnv, GDPvalState, GDPvalStepResult

__all__ = [
    "GDPvalEnv",
    "AsyncGDPvalEnv",
    "GDPvalState",
    "GDPvalStepResult",
]
