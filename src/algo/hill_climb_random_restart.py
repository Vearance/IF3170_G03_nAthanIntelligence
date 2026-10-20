from __future__ import annotations

from collections.abc import Callable
from random import Random
from time import perf_counter

from src.algo.hill_climb import steepest_ascent_hill_climbing
from src.models.result import Result
from src.models.state import State
from src.utils.evaluate import Evaluate
from src.utils.random_state import random_initial_state


def random_restart_hill_climbing(
    initial_state: State,
    *,
    max_restarts: int = 10,
    seed: int | None = None,
    on_iteration: Callable[[State, int], None] | None = None,
) -> Result:
    if isinstance(max_restarts, bool) or not isinstance(max_restarts, int):
        raise TypeError("max_restarts harus berupa integer")
    if max_restarts < 0:
        raise ValueError("max_restarts tidak boleh negatif")

    started_at = perf_counter()
    initial = initial_state.copy()
    if not Evaluate.is_valid(initial):
        raise ValueError("initial_state harus valid")
    rng = Random(seed)
    best = initial.copy()
    iterations = 0
    value_history = [initial.value]
    iterations_per_restart: list[int] = []
    restart_histories: list[list[int]] = []

    def report(state: State, iteration: int) -> None:
        if on_iteration is not None:
            on_iteration(state, iterations + iteration)

    for restart in range(max_restarts + 1):
        current = initial if restart == 0 else random_initial_state(initial.problem, rng)
        result = steepest_ascent_hill_climbing(
            current, on_iteration=report if on_iteration is not None else None
        )
        iterations += result.iterations
        iterations_per_restart.append(result.iterations)
        restart_histories.append(result.value_history)
        value_history.extend(result.value_history[1:])
        if result.state.value > best.value:
            best = result.state.copy()

    return Result(
        initial_state=initial,
        state=best,
        value_history=value_history,
        iterations=iterations,
        duration_seconds=perf_counter() - started_at,
        metrics={
            "restart_count": max_restarts,
            "max_restarts": max_restarts,
            "iterations_per_restart": iterations_per_restart,
            "restart_histories": restart_histories,
        },
    )
