from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.models.problem import Problem
from src.models.result import Result
from src.models.state import State
from src.models.types.orientation import Orientation
from src.utils.parser import problem_from_dict, problem_to_dict


def state_to_snapshot_dict(state: State) -> dict[str, Any]:
    return {
        "grid": state.grid,
        "pkg_orientation": {
            package_id: {
                "w": orientation.w,
                "l": orientation.l,
                "h": orientation.h,
            }
            for package_id, orientation in state.pkg_orientation.items()
        },
        "out_pkg": sorted(state.out_pkg),
        "value": state.value,
    }


def state_from_snapshot_dict(
    problem: Problem,
    data: dict[str, Any],
) -> State:
    if not isinstance(data, dict):
        raise TypeError("state snapshot harus berupa object")

    grid = data.get("grid")
    orientations_data = data.get("pkg_orientation")
    out_pkg = data.get("out_pkg")
    if not isinstance(grid, list):
        raise TypeError("state snapshot harus memiliki grid berupa array")
    if not isinstance(orientations_data, dict):
        raise TypeError(
            "state snapshot harus memiliki pkg_orientation berupa object"
        )
    if not isinstance(out_pkg, list) or not all(
        isinstance(package_id, str) for package_id in out_pkg
    ):
        raise TypeError("state snapshot harus memiliki out_pkg berupa array string")

    orientations: dict[str, Orientation] = {}
    for package_id in problem.packages:
        orientation_data = orientations_data.get(package_id)
        if not isinstance(orientation_data, dict):
            raise TypeError(
                f"orientation package tidak ditemukan di snapshot: {package_id}"
            )
        try:
            orientations[package_id] = Orientation(
                w=orientation_data["w"],
                l=orientation_data["l"],
                h=orientation_data["h"],
            )
        except KeyError as err:
            raise ValueError(
                f"orientation package tidak lengkap di snapshot: {package_id}"
            ) from err

    return State(
        problem=problem,
        grid=grid,
        pkg_orientation=orientations,
        out_pkg=set(out_pkg),
        value=data.get("value", 0),
    )


def snapshot_to_dict(result: Result, algorithm: str) -> dict[str, Any]:
    if not isinstance(result, Result):
        raise TypeError("result harus berupa instance Result")

    return {

        "algorithm": algorithm,
        "problem": problem_to_dict(result.initial_state.problem),
        "initial_state": state_to_snapshot_dict(result.initial_state),
        "final_state": state_to_snapshot_dict(result.state),
    }


def write_snapshot(
    path: str | Path,
    result: Result,
    algorithm: str,
) -> Path:
    snapshot_path = Path(path)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(
        json.dumps(snapshot_to_dict(result, algorithm), indent=2),
        encoding="utf-8",
    )
    return snapshot_path


def read_snapshot(path: str | Path) -> tuple[str, State, State]:
    snapshot_path = Path(path)
    with snapshot_path.open(encoding="utf-8") as snapshot_file:
        data = json.load(snapshot_file)

    if not isinstance(data, dict):
        raise TypeError("snapshot harus berupa object")

    algorithm = data.get("algorithm")
    if not isinstance(algorithm, str) or not algorithm:
        raise ValueError("snapshot harus memiliki algorithm berupa string")

    problem_data = data.get("problem")
    initial_data = data.get("initial_state")
    final_data = data.get("final_state")
    if not isinstance(problem_data, dict):
        raise TypeError("snapshot harus memiliki problem berupa object")
    if not isinstance(initial_data, dict) or not isinstance(final_data, dict):
        raise TypeError(
            "snapshot harus memiliki initial_state dan final_state berupa object"
        )

    problem = problem_from_dict(problem_data)
    return (
        algorithm,
        state_from_snapshot_dict(problem, initial_data),
        state_from_snapshot_dict(problem, final_data),
    )


def resolve_snapshot_file(
    snapshot_file: str | Path,
    output_directory: str | Path,
) -> Path:
    path = Path(snapshot_file).expanduser()
    if path.is_file():
        return path.resolve()

    if path.parent == Path("."):
        output_path = Path(output_directory) / path.name
        if output_path.is_file():
            return output_path.resolve()

    raise FileNotFoundError(f"snapshot JSON tidak ditemukan: {snapshot_file}")
