from __future__ import annotations

from collections.abc import Callable

from src.models.result import Result
from src.models.state import State

# TODO: placeholder
def genetic_algorithm(
    initial_state: State,
    *,
    on_iteration: Callable[[State, int], None] | None = None,
) -> Result:
	return Result(
        initial_state=None,
        state=None,
        value_history=None,
        iterations=None,
        duration_seconds=None,
        metrics=None,
    )
