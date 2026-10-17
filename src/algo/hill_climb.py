from __future__ import annotations

from collections.abc import Callable
from time import perf_counter

from src.models.result import Result
from src.models.state import State
from src.utils.evaluate import Evaluate
from src.utils.move import neighbors


def steepest_ascent_hill_climbing(
    initial_state: State,
    *,
    max_iterations: int | None = None,
    on_iteration: Callable[[State, int], None] | None = None,
) -> Result:
    if max_iterations is not None and (
        isinstance(max_iterations, bool) or not isinstance(max_iterations, int)
    ):
        raise TypeError("max_iterations harus berupa integer atau None")
    if max_iterations is not None and max_iterations < 0:
        raise ValueError("max_iterations tidak boleh negatif")

    started_at = perf_counter()
    current = initial_state.copy()
    if not Evaluate.is_valid(current):
        raise ValueError("initial_state harus valid")

    current.value = Evaluate.value(current)
    initial = current.copy()
    value_history = [current.value]
    iterations = 0
    neighbor_states_evaluated = 0
    termination_reason = "local_optimum"

    while max_iterations is None or iterations < max_iterations:
        best_neighbor: State | None = None
        for candidate in neighbors(current):
            neighbor_states_evaluated += 1
            if candidate.value > current.value and (
                best_neighbor is None or candidate.value > best_neighbor.value
            ):
                best_neighbor = candidate

        iterations += 1
        if best_neighbor is None:
            value_history.append(current.value)
            if on_iteration is not None:
                on_iteration(current.copy(), iterations)
            break

        current = best_neighbor
        value_history.append(current.value)
        if on_iteration is not None:
            on_iteration(current.copy(), iterations)
    else:
        termination_reason = "max_iterations"

    return Result(
        initial_state=initial,
        state=current.copy(),
        value_history=value_history,
        iterations=iterations,
        duration_seconds=perf_counter() - started_at,
        metrics={
            "termination_reason": termination_reason,
            "neighbor_states_evaluated": neighbor_states_evaluated,
        },
    )
