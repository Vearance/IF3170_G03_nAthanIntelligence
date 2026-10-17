from __future__ import annotations

import math
import random
from collections.abc import Callable
from time import perf_counter

from src.models.package import Package
from src.models.result import Result
from src.models.state import State
from src.models.types.orientation import Orientation
from src.utils.evaluate import Evaluate


def simulated_annealing(
    initial_state: State,
    *,
    initial_temperature: float = 100.0,
    cooling_rate: float = 0.995,
    minimum_temperature: float = 0.1,
    max_iterations: int = 10_000,
    seed: int | None = None,
    on_iteration: Callable[[State, int], None] | None = None,
) -> Result:
    if initial_temperature <= 0:
        raise ValueError("initial_temperature harus lebih besar dari nol")
    if not 0 < cooling_rate < 1:
        raise ValueError("cooling_rate harus berada di antara 0 dan 1")
    if minimum_temperature <= 0:
        raise ValueError("minimum_temperature harus lebih besar dari nol")
    if max_iterations < 0:
        raise ValueError("max_iterations tidak boleh negatif")

    started_at = perf_counter()
    rng = random.Random(seed)
    current = initial_state.copy()
    if not Evaluate.is_valid(current):
        current = State.empty(initial_state.problem)
    initial = current.copy()
    best = current.copy()
    value_history = [best.value]
    temperature = initial_temperature
    accepted_moves = 0
    iterations = 0

    while iterations < max_iterations and temperature > minimum_temperature:
        candidate = _random_neighbour(current, rng)
        if candidate is None:
            break

        delta = candidate.value - current.value
        if delta >= 0 or rng.random() < math.exp(delta / temperature):
            current = candidate
            accepted_moves += 1
            if current.value > best.value:
                best = current.copy()

        temperature *= cooling_rate
        iterations += 1
        value_history.append(best.value)
        if on_iteration is not None:
            on_iteration(current.copy(), iterations)

    return Result(
        initial_state=initial,
        state=best,
        value_history=value_history,
        iterations=iterations,
        duration_seconds=perf_counter() - started_at,
        metrics={"accepted_moves": accepted_moves},
    )

def _random_neighbour(state: State, rng: random.Random) -> State | None:
    operations = ["place"] if state.out_pkg else []
    placed = sorted(state.package_ids_in_grid())
    if placed:
        operations.extend(("remove", "rotate"))
    if not operations:
        return None

    for _ in range(8):
        operation = rng.choice(operations)
        if operation == "place":
            candidate = _try_place(state, rng)
        elif operation == "remove":
            candidate = _remove(state, rng.choice(placed))
        else:
            candidate = _try_rotate(state, rng.choice(placed), rng)
        if candidate is not None:
            return candidate
    return None

def _try_place(state: State, rng: random.Random) -> State | None:
    return _place(state, rng.choice(sorted(state.out_pkg)), rng)

def _place(state: State, package_id: str, rng: random.Random) -> State | None:
    package = state.problem.get_package(package_id)
    orientation = state.pkg_orientation[package_id]
    width, length, height = _oriented_dimensions(package, orientation)
    truck = state.problem.truck.dimensions
    max_x = truck.w - width
    max_y = truck.l - length
    max_z = truck.h - height
    if min(max_x, max_y, max_z) < 0:
        return None

    z_levels = {0}
    for layer_index, layer in enumerate(state.grid):
        if layer_index < max_z and any(
            cell is not None for row in layer for cell in row
        ):
            z_levels.add(layer_index + 1)
    z_levels = tuple(sorted(z_levels))

    for _ in range(24):
        x = rng.randrange(max_x + 1)
        y = rng.randrange(max_y + 1)
        z = rng.choice(z_levels)
        if any(
            state.grid[current_z][current_y][current_x] is not None
            for current_z in range(z, z + height)
            for current_y in range(y, y + length)
            for current_x in range(x, x + width)
        ):
            continue

        candidate = state.copy()
        for current_z in range(z, z + height):
            for current_y in range(y, y + length):
                for current_x in range(x, x + width):
                    candidate.grid[current_z][current_y][current_x] = package_id

        candidate.out_pkg.discard(package_id)
        candidate.value = state.value + package.value
        if Evaluate.is_valid(candidate):
            return candidate
    return None

def _remove(state: State, package_id: str) -> State:
    candidate = state.copy()
    for layer in candidate.grid:
        for row in layer:
            for index, cell in enumerate(row):
                if cell == package_id:
                    row[index] = None
    candidate.out_pkg.add(package_id)
    candidate.value = state.value - state.problem.get_package(package_id).value
    return candidate

def _try_rotate(
    state: State, package_id: str, rng: random.Random
) -> State | None:
    orientation = state.pkg_orientation[package_id]
    new_orientation = orientation.rotate(rng.choice(("x", "y", "z")))
    if new_orientation == orientation:
        return None

    candidate = _remove(state, package_id)
    candidate.pkg_orientation[package_id] = new_orientation
    return _place(candidate, package_id, rng)


def _oriented_dimensions(
    package: Package, orientation: Orientation
) -> tuple[int, int, int]:
    dims = [
        getattr(package.dimensions, dimension)
        for axis in ("x", "y", "z")
        for dimension in ("w", "l", "h")
        if getattr(orientation, dimension) == axis
    ]
    return dims[0], dims[1], dims[2]