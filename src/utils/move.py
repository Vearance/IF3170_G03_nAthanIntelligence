from collections.abc import Iterator
from itertools import combinations

from src.models.package import Package
from src.models.state import State
from src.models.types.orientation import Orientation
from src.models.types.position import Position
from src.utils.evaluate import Evaluate


def try_move(
    state: State,
    package_id: str,
    position: Position | None,
) -> State | None:
    if package_id not in state.problem.packages:
        return None

    current_position = _package_anchor(state, package_id)
    if current_position == position:
        return None

    candidate = state.copy()
    try:
        _remove_from_grid(candidate, package_id)
        candidate.out_pkg.discard(package_id)

        if position is None:
            candidate.out_pkg.add(package_id)
        else:
            _place_on_grid(candidate, package_id, position)

        return _valid_candidate(candidate)
    except (IndexError, KeyError, TypeError, ValueError):
        return None


def try_rotate(
    state: State,
    package_id: str,
    axis: str,
) -> State | None:
    if package_id not in state.problem.packages:
        return None

    candidate = state.copy()
    try:
        position = _package_anchor(candidate, package_id)
        candidate.pkg_orientation[package_id] = candidate.pkg_orientation[
            package_id
        ].rotate(axis)
        if position is not None:
            _remove_from_grid(candidate, package_id)
            _place_on_grid(candidate, package_id, position)

        return _valid_candidate(candidate)
    except (IndexError, KeyError, TypeError, ValueError):
        return None


def try_swap(
    state: State,
    first_package_id: str,
    second_package_id: str,
) -> State | None:
    if (
        first_package_id == second_package_id
        or first_package_id not in state.problem.packages
        or second_package_id not in state.problem.packages
    ):
        return None

    first_position = _package_anchor(state, first_package_id)
    second_position = _package_anchor(state, second_package_id)
    if first_position is None and second_position is None:
        return None

    candidate = state.copy()
    try:
        _remove_from_grid(candidate, first_package_id)
        _remove_from_grid(candidate, second_package_id)
        candidate.out_pkg.discard(first_package_id)
        candidate.out_pkg.discard(second_package_id)

        if second_position is None:
            candidate.out_pkg.add(first_package_id)
        else:
            _place_on_grid(candidate, first_package_id, second_position)

        if first_position is None:
            candidate.out_pkg.add(second_package_id)
        else:
            _place_on_grid(candidate, second_package_id, first_position)

        return _valid_candidate(candidate)
    except (IndexError, KeyError, TypeError, ValueError):
        return None


def try_place(
    state: State,
    package_id: str,
    position: Position,
) -> State | None:
    return try_move(state, package_id, position)


def first_valid_placement(state: State, package_id: str) -> State | None:
    if package_id not in state.out_pkg:
        return None

    package = state.problem.get_package(package_id)
    truck = state.problem.truck.dimensions
    for orientation in _orientation_variants(state.pkg_orientation[package_id]):
        width, length, height = _oriented_dimensions(package, orientation)
        for z in range(truck.h - height + 1):
            for y in range(truck.l - length + 1):
                for x in range(truck.w - width + 1):
                    position = Position(x=x, y=y, z=z)
                    if orientation == state.pkg_orientation[package_id]:
                        candidate = try_place(state, package_id, position)
                    else:
                        candidate = _try_place_with_orientation(
                            state,
                            package_id,
                            position,
                            orientation,
                        )
                    if candidate is not None:
                        return candidate
    return None


def try_remove(state: State, package_id: str) -> State | None:
    return try_move(state, package_id, None)


def neighbors(state: State) -> Iterator[State]:
    seen = {_state_key(state)}
    package_ids = sorted(state.problem.packages)
    truck = state.problem.truck.dimensions

    for package_id in package_ids:
        position = _package_anchor(state, package_id)
        if position is None:
            package = state.problem.get_package(package_id)
            for orientation in _orientation_variants(state.pkg_orientation[package_id]):
                width, length, height = _oriented_dimensions(package, orientation)
                for z in range(truck.h - height + 1):
                    for y in range(truck.l - length + 1):
                        for x in range(truck.w - width + 1):
                            next_position = Position(x=x, y=y, z=z)
                            if orientation == state.pkg_orientation[package_id]:
                                candidate = try_place(state, package_id, next_position)
                            else:
                                candidate = _try_place_with_orientation(
                                    state,
                                    package_id,
                                    next_position,
                                    orientation,
                                )
                            if candidate is not None:
                                candidate_key = _state_key(candidate)
                                if candidate_key not in seen:
                                    seen.add(candidate_key)
                                    yield candidate
            continue

        for axis in ("x", "y", "z"):
            candidate = try_rotate(state, package_id, axis)
            if candidate is not None:
                candidate_key = _state_key(candidate)
                if candidate_key not in seen:
                    seen.add(candidate_key)
                    yield candidate

        for dx, dy, dz in (
            (-1, 0, 0),
            (1, 0, 0),
            (0, -1, 0),
            (0, 1, 0),
            (0, 0, -1),
            (0, 0, 1),
        ):
            x = position.x + dx
            y = position.y + dy
            z = position.z + dz
            if min(x, y, z) < 0:
                continue

            candidate = try_move(
                state,
                package_id,
                Position(x=x, y=y, z=z),
            )
            if candidate is not None:
                candidate_key = _state_key(candidate)
                if candidate_key not in seen:
                    seen.add(candidate_key)
                    yield candidate

        candidate = try_remove(state, package_id)
        if candidate is not None:
            candidate_key = _state_key(candidate)
            if candidate_key not in seen:
                seen.add(candidate_key)
                yield candidate

    for first_package_id, second_package_id in combinations(package_ids, 2):
        candidate = try_swap(state, first_package_id, second_package_id)
        if candidate is not None:
            candidate_key = _state_key(candidate)
            if candidate_key not in seen:
                seen.add(candidate_key)
                yield candidate


def _try_place_with_orientation(
    state: State,
    package_id: str,
    position: Position,
    orientation: Orientation,
) -> State | None:
    candidate = state.copy()
    try:
        candidate.pkg_orientation[package_id] = orientation
        _place_on_grid(candidate, package_id, position)
        candidate.out_pkg.discard(package_id)
        return _valid_candidate(candidate)
    except (IndexError, KeyError, TypeError, ValueError):
        return None


def _orientation_variants(orientation: Orientation) -> list[Orientation]:
    variants = [orientation]
    seen = {(orientation.w, orientation.l, orientation.h)}
    index = 0

    while index < len(variants):
        current = variants[index]
        index += 1
        for axis in ("x", "y", "z"):
            rotated = current.rotate(axis)
            key = (rotated.w, rotated.l, rotated.h)
            if key not in seen:
                seen.add(key)
                variants.append(rotated)

    return variants


def _valid_candidate(state: State) -> State | None:
    state.value = Evaluate.value(state)
    if Evaluate.is_valid(state):
        return state
    return None


def _remove_from_grid(state: State, package_id: str) -> None:
    for layer in state.grid:
        for row in layer:
            for x, cell in enumerate(row):
                if cell == package_id:
                    row[x] = None


def _place_on_grid(state: State, package_id: str, position: Position) -> None:
    package = state.problem.get_package(package_id)
    orientation = state.pkg_orientation[package_id]
    width, length, height = _oriented_dimensions(package, orientation)
    dimensions = state.problem.truck.dimensions

    if (
        position.x + width > dimensions.w
        or position.y + length > dimensions.l
        or position.z + height > dimensions.h
    ):
        raise IndexError(f"package {package_id} berada di luar truck")

    cells = [
        (x, y, z)
        for z in range(position.z, position.z + height)
        for y in range(position.y, position.y + length)
        for x in range(position.x, position.x + width)
    ]
    if any(state.grid[z][y][x] is not None for x, y, z in cells):
        raise ValueError(f"package {package_id} bertabrakan dengan package lain")

    for x, y, z in cells:
        state.grid[z][y][x] = package_id


def _package_anchor(state: State, package_id: str) -> Position | None:
    cells = [
        (x, y, z)
        for z, layer in enumerate(state.grid)
        for y, row in enumerate(layer)
        for x, cell in enumerate(row)
        if cell == package_id
    ]
    if not cells:
        return None

    return Position(
        x=min(x for x, _, _ in cells),
        y=min(y for _, y, _ in cells),
        z=min(z for _, _, z in cells),
    )


def _oriented_dimensions(
    package: Package,
    orientation: Orientation,
) -> tuple[int, int, int]:
    dimensions_by_axis = {
        orientation.w: package.dimensions.w,
        orientation.l: package.dimensions.l,
        orientation.h: package.dimensions.h,
    }
    return (
        dimensions_by_axis["x"],
        dimensions_by_axis["y"],
        dimensions_by_axis["z"],
    )


def _state_key(state: State) -> tuple[object, ...]:
    return (
        tuple(tuple(tuple(row) for row in layer) for layer in state.grid),
        tuple(
            (package_id, orientation.w, orientation.l, orientation.h)
            for package_id, orientation in sorted(state.pkg_orientation.items())
        ),
        tuple(sorted(state.out_pkg)),
    )
