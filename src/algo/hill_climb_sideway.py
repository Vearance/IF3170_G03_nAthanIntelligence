from __future__ import annotations

from collections.abc import Callable
from random import Random
from time import perf_counter

from src.models.result import Result
from src.models.state import State
from src.utils.evaluate import Evaluate
from src.utils.move import neighbors


def hill_climbing_with_sideways_move(
    initial_state: State,
    *,
    max_sideways_moves: int = 100,
    seed: int | None = None,
    on_iteration: Callable[[State, int], None] | None = None,
) -> Result:
    if isinstance(max_sideways_moves, bool) or not isinstance(max_sideways_moves, int):
        raise TypeError("max_sideways_moves harus berupa integer")
    if max_sideways_moves < 0:
        raise ValueError("max_sideways_moves tidak boleh negatif")

    started_at = perf_counter()
    current = initial_state.copy()
    if not Evaluate.is_valid(current):
        raise ValueError("initial_state harus valid")

    current.value = Evaluate.value(current)
    initial = current.copy()
    value_history = [current.value]
    iterations = 0
    sideways_moves = 0
    rng = Random(seed)

    while True:
        best_value = current.value
        best_neighbors: list[State] = []
        for candidate in neighbors(current):
            if candidate.value > best_value:
                best_value = candidate.value
                best_neighbors = [candidate]
            elif candidate.value == best_value:
                best_neighbors.append(candidate)

        iterations += 1
        if not best_neighbors or (
            best_value == current.value and sideways_moves >= max_sideways_moves
        ):
            value_history.append(current.value)
            if on_iteration is not None:
                on_iteration(current.copy(), iterations)
            break

        best_neighbor = rng.choice(best_neighbors)
        if best_neighbor.value == current.value:
            sideways_moves += 1
        current = best_neighbor
        value_history.append(current.value)
        if on_iteration is not None:
            on_iteration(current.copy(), iterations)
        if max_sideways_moves > 0 and sideways_moves >= max_sideways_moves:
            break

    return Result(
        initial_state=initial,
        state=current.copy(),
        value_history=value_history,
        iterations=iterations,
        duration_seconds=perf_counter() - started_at,
        metrics={
            "sideways_moves": sideways_moves,
            "max_sideways_moves": max_sideways_moves,
        },
    )
