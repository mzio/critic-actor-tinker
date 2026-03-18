"""
Environments
"""

from typing import Any

from .base import Environment
from .types import EnvironmentState, EnvironmentStateWithAnswer, EnvironmentStepResult


def get_env(name: str, is_async: bool = True, **kwargs: Any) -> Environment:
    """
    Get environment based on name
    """
    if name == "snorkel_finance":
        if is_async:
            from .snorkel_finance import AsyncSnorkelFinanceEnv

            return AsyncSnorkelFinanceEnv(**kwargs)
        else:
            from .snorkel_finance import SnorkelFinanceEnv

            return SnorkelFinanceEnv(**kwargs)

    elif name == "snorkel_insurance":
        if is_async:
            from .snorkel_insurance import AsyncSnorkelInsuranceEnv

            return AsyncSnorkelInsuranceEnv(**kwargs)
        else:
            from .snorkel_insurance import SnorkelInsuranceEnv

            return SnorkelInsuranceEnv(**kwargs)

    elif name == "gdpval":
        if is_async:
            from .gdpval import AsyncGDPvalEnv

            return AsyncGDPvalEnv(**kwargs)
        else:
            from .gdpval import GDPvalEnv

            return GDPvalEnv(**kwargs)

    elif name == "browsecomp_plus_search":
        if is_async:
            from .browsecomp_plus import AsyncBrowseCompPlusSearchEnv

            return AsyncBrowseCompPlusSearchEnv(**kwargs)
        else:
            from .browsecomp_plus import BrowseCompPlusSearchEnv

            return BrowseCompPlusSearchEnv(**kwargs)

    raise NotImplementedError(f"Sorry invalid environment: '{name}'.")


def load_env(name: str, **kwargs: Any) -> Environment:
    """
    Alias for get_env
    """
    return get_env(name, **kwargs)


__all__ = [
    "get_env",
    "load_env",
    "Environment",
    "EnvironmentState",
    "EnvironmentStateWithAnswer",
    "EnvironmentStepResult",
]
