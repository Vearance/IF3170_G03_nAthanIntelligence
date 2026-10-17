from __future__ import annotations

from collections.abc import Callable, Iterator
from itertools import combinations
from time import perf_counter

from src.models.result import Result
from src.models.state import State
from src.utils.evaluate import Evaluate
from src.utils.move import first_valid_placement, try_swap


def steepest_ascent_hill_climbing(
    initial_state: State,
    *,
    on_iteration: Callable[[State, int], None] | None = None,
) -> Result:
    started_at = perf_counter()
    current = initial_state.copy()
    if not Evaluate.is_valid(current):
        raise ValueError("initial_state harus valid")

    current.value = Evaluate.value(current)
    initial = current.copy()
    value_history = [current.value]
    iterations = 0
    neighbor_states_evaluated = 0

    while True:
        best_neighbor: State | None = None
        for candidate in _improving_neighbors(current):
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

    return Result(
        initial_state=initial,
        state=current.copy(),
        value_history=value_history,
        iterations=iterations,
        duration_seconds=perf_counter() - started_at,
        metrics={
            "termination_reason": "local_optimum",
            "neighbor_states_evaluated": neighbor_states_evaluated,
        },
    )


def _improving_neighbors(state: State) -> Iterator[State]:
    package_ids = sorted(state.problem.packages)

    for package_id in package_ids:
        if package_id not in state.out_pkg:
            continue

        package = state.problem.get_package(package_id)
        if package.value <= 0:
            continue

        candidate = first_valid_placement(state, package_id)
        if candidate is not None and candidate.value > state.value:
            yield candidate

    for first_package_id, second_package_id in combinations(package_ids, 2):
        first_is_out = first_package_id in state.out_pkg
        second_is_out = second_package_id in state.out_pkg
        if first_is_out == second_is_out:
            continue

        if first_is_out:
            added_package_id, removed_package_id = (
                first_package_id,
                second_package_id,
            )
        else:
            added_package_id, removed_package_id = (
                second_package_id,
                first_package_id,
            )

        added = state.problem.get_package(added_package_id)
        removed = state.problem.get_package(removed_package_id)
        if added.value <= removed.value:
            continue

        candidate = try_swap(state, first_package_id, second_package_id)
        if candidate is not None and candidate.value > state.value:
            yield candidate
