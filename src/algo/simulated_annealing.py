from __future__ import annotations

import math
import random
from dataclasses import dataclass

from src.models.package import Package
from src.models.state import State
from src.models.types.orientation import Orientation
from src.utils.evaluate import Evaluate


@dataclass(frozen=True)
class AnnealingResult:
    state: State
    iterations: int
    accepted_moves: int


def simulated_annealing(
    initial_state: State,
    *,
    initial_temperature: float = 100.0,
    cooling_rate: float = 0.995,
    minimum_temperature: float = 0.1,
    max_iterations: int = 10_000,
    seed: int | None = None,
) -> AnnealingResult:
    if initial_temperature <= 0:
        raise ValueError("initial_temperature harus lebih besar dari nol")
    if not 0 < cooling_rate < 1:
        raise ValueError("cooling_rate harus berada di antara 0 dan 1")
    if minimum_temperature <= 0:
        raise ValueError("minimum_temperature harus lebih besar dari nol")
    if max_iterations < 0:
        raise ValueError("max_iterations tidak boleh negatif")

    rng = random.Random(seed)
    current = initial_state.copy()
    if not Evaluate.is_valid(current):
        current = State.empty(initial_state.problem)
    best = current.copy()
    temperature = initial_temperature
    accepted_moves = 0
    iterations = 0

    while iterations < max_iterations and temperature > minimum_temperature:
        candidate = _random_neighbour(current, rng)
        if candidate is None:
            break

        delta = Evaluate.value(candidate) - Evaluate.value(current)
        if delta >= 0 or rng.random() < math.exp(delta / temperature):
            current = candidate
            accepted_moves += 1
            if Evaluate.value(current) > Evaluate.value(best):
                best = current.copy()

        temperature *= cooling_rate
        iterations += 1

    if not Evaluate.is_valid(best):
        raise RuntimeError("simulated annealing menghasilkan state tidak valid")

    return AnnealingResult(
        state=best,
        iterations=iterations,
        accepted_moves=accepted_moves,
    )


def _random_neighbour(state: State, rng: random.Random) -> State | None:
    operations = ["place"] if state.out_pkg else []
    placed = list(state.package_ids_in_grid())
    if placed:
        operations.extend(("remove", "rotate"))
    if not operations:
        return None

    for _ in range(25):
        operation = rng.choice(operations)
        if operation == "place":
            candidate = _try_place(state, rng)
        elif operation == "remove":
            candidate = _remove(state, rng.choice(placed))
        else:
            candidate = _try_rotate(state, rng.choice(placed), rng)
        if candidate is not None and Evaluate.is_valid(candidate):
            return candidate
    return None


def _try_place(state: State, rng: random.Random) -> State | None:
    package_id = rng.choice(tuple(state.out_pkg))
    package = state.problem.get_package(package_id)
    orientation = state.pkg_orientation[package_id]
    width, length, height = _oriented_dimensions(package, orientation)
    truck = state.problem.truck.dimensions
    positions = [
        (x, y, z)
        for z in range(truck.h - height + 1)
        for y in range(truck.l - length + 1)
        for x in range(truck.w - width + 1)
    ]
    rng.shuffle(positions)

    for x, y, z in positions:
        candidate = state.copy()
        if any(
            candidate.grid[current_z][current_y][current_x] is not None
            for current_z in range(z, z + height)
            for current_y in range(y, y + length)
            for current_x in range(x, x + width)
        ):
            continue
        for current_z in range(z, z + height):
            for current_y in range(y, y + length):
                for current_x in range(x, x + width):
                    candidate.grid[current_z][current_y][current_x] = package_id
        candidate.out_pkg.remove(package_id)
        candidate.value = candidate.recalculate_value()
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
    candidate.value = candidate.recalculate_value()
    return candidate


def _try_rotate(
    state: State, package_id: str, rng: random.Random
) -> State | None:
    package = state.problem.get_package(package_id)
    old_orientation = state.pkg_orientation[package_id]
    new_orientation = old_orientation.rotate(rng.choice(("x", "y", "z")))
    if new_orientation == old_orientation:
        return None

    candidate = _remove(state, package_id)
    candidate.pkg_orientation[package_id] = new_orientation
    return _try_place_existing(candidate, package_id, package, rng)


def _try_place_existing(
    state: State, package_id: str, package: Package, rng: random.Random
) -> State | None:
    orientation = state.pkg_orientation[package_id]
    width, length, height = _oriented_dimensions(package, orientation)
    truck = state.problem.truck.dimensions
    positions = [
        (x, y, z)
        for z in range(truck.h - height + 1)
        for y in range(truck.l - length + 1)
        for x in range(truck.w - width + 1)
    ]
    rng.shuffle(positions)
    for x, y, z in positions:
        candidate = state.copy()
        if any(
            candidate.grid[current_z][current_y][current_x] is not None
            for current_z in range(z, z + height)
            for current_y in range(y, y + length)
            for current_x in range(x, x + width)
        ):
            continue
        for current_z in range(z, z + height):
            for current_y in range(y, y + length):
                for current_x in range(x, x + width):
                    candidate.grid[current_z][current_y][current_x] = package_id
        candidate.out_pkg.remove(package_id)
        candidate.value = candidate.recalculate_value()
        return candidate
    return None


def _oriented_dimensions(
    package: Package, orientation: Orientation
) -> tuple[int, int, int]:
    dimensions: list[int] = []
    for axis in ("x", "y", "z"):
        for dimension in ("w", "l", "h"):
            if getattr(orientation, dimension) == axis:
                dimensions.append(getattr(package.dimensions, dimension))
                break
    return dimensions[0], dimensions[1], dimensions[2]
