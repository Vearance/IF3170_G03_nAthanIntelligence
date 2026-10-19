from __future__ import annotations

from random import Random

from src.models.problem import Problem
from src.models.state import State
from src.models.types.orientation import VALID_ORIENTATION, Orientation
from src.models.types.position import Position
from src.utils.move import (
    _oriented_dimensions,
    _package_anchor,
    try_place,
)


def random_position(
    state: State,
    package_id: str,
    rng: Random,
) -> Position | None:
    package = state.problem.get_package(package_id)
    width, length, height = _oriented_dimensions(
        package,
        state.pkg_orientation[package_id],
    )

    truck = state.problem.truck.dimensions
    max_x = truck.w - width
    max_y = truck.l - length
    max_z = truck.h - height
    if min(max_x, max_y, max_z) < 0:
        return None

    z_levels = {0}
    for support_id in sorted(state.package_ids_in_grid()):
        support = state.problem.get_package(support_id)
        if support_id == package_id or support.isFragile:
            continue

        anchor = _package_anchor(state, support_id)
        if anchor is None:
            continue

        support_height = _oriented_dimensions(
            support,
            state.pkg_orientation[support_id],
        )[2]
        top = anchor.z + support_height
        if top <= max_z:
            z_levels.add(top)

    return Position(
        x=rng.randint(0, max_x),
        y=rng.randint(0, max_y),
        z=rng.choice(sorted(z_levels)),
    )


def random_initial_state(problem: Problem, rng: Random) -> State:
    state = State.empty(problem)
    package_ids = sorted(problem.packages)
    rng.shuffle(package_ids)

    for package_id in package_ids:
        if rng.random() < 0.2:
            continue

        orientations = [Orientation(*axes) for axes in sorted(VALID_ORIENTATION)]
        rng.shuffle(orientations)

        for orientation in orientations:
            state.pkg_orientation[package_id] = orientation
            for _ in range(5):
                position = random_position(state, package_id, rng)
                if position is None:
                    break

                candidate = try_place(state, package_id, position)
                if candidate is not None:
                    state = candidate
                    break

            if package_id not in state.out_pkg:
                break

    return state
