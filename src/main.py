from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from random import Random
from typing import Any

from src.algo.genetic import genetic_algorithm
from src.algo.hill_climb import steepest_ascent_hill_climbing
from src.algo.simulated_annealing import simulated_annealing
from src.models.result import Result
from src.models.state import State
from src.models.types.orientation import Orientation
from src.utils.evaluate import Evaluate
from src.utils.parser import json_to_problem, resolve_problem_file
from src.utils.random_state import random_initial_state
from src.utils.snapshot import write_snapshot

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIRECTORY = REPOSITORY_ROOT / "test" / "input"
DEFAULT_OUTPUT_DIRECTORY = REPOSITORY_ROOT / "output"
DEFAULT_INPUT_FILE = Path("test/input/test-1.json")
SNAPSHOT_ALIASES = {
    "hill-climbing": "hc",
    "simulated-annealing": "sa",
    "genetic-algorithm": "ga",
}


def _non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as err:
        raise argparse.ArgumentTypeError("harus berupa integer") from err
    if parsed < 0:
        raise argparse.ArgumentTypeError("tidak boleh negatif")
    return parsed



def _cells_by_package(state: State) -> dict[str, list[tuple[int, int, int]]]:
    cells_by_package: dict[str, list[tuple[int, int, int]]] = {}
    for z, layer in enumerate(state.grid):
        for y, row in enumerate(layer):
            for x, package_id in enumerate(row):
                if package_id is not None:
                    cells_by_package.setdefault(package_id, []).append((x, y, z))
    return cells_by_package


def _orientation_to_dict(orientation: Orientation) -> dict[str, str]:
    return {"w": orientation.w, "l": orientation.l, "h": orientation.h}




def _format_state_summary(state: State) -> list[str]:
    cells_by_package = _cells_by_package(state)
    lines = [f"value: {state.value}"]
    for package_id in state.problem.packages:
        cells = cells_by_package.get(package_id)
        if not cells:
            lines.append(f"- {package_id}: outside truck")
            continue
        x = min(cell[0] for cell in cells)
        y = min(cell[1] for cell in cells)
        z = min(cell[2] for cell in cells)
        orientation = _orientation_to_dict(state.pkg_orientation[package_id])
        lines.append(
            f"- {package_id}: position=({x}, {y}, {z}), orientation={orientation}"
        )
    return lines


def _format_details(
    input_path: Path,
    algorithm_name: str,
    result: Result,
) -> str:
    final_state = result.state
    placed_count = len(final_state.package_ids_in_grid())
    total_packages = len(final_state.problem.packages)
    metrics = result.metrics or {}

    lines = [
        "Search Result",
        "===================================",
        f"input: {input_path}",
        f"algorithm: {algorithm_name}",
        f"iterations: {result.iterations}",
        f"duration_seconds: {result.duration_seconds:.6f}",
        f"packages_placed: {placed_count} / {total_packages}",
        f"final_value: {final_state.value}",
        f"final_state_valid: {Evaluate.is_valid(final_state)}",
        "",
        "parameters and metrics:",
    ]
    if metrics:
        lines.extend(f"- {name}: {value}" for name, value in metrics.items())
    else:
        lines.append("- none")

    lines.extend(["", "initial state:"])
    lines.extend(_format_state_summary(result.initial_state))
    lines.extend(["", "final state:"])
    lines.extend(_format_state_summary(final_state))
    lines.extend(["", "objective history:"])
    lines.extend(
        f"- iteration {iteration}: {value}"
        for iteration, value in enumerate(result.value_history)
    )
    return "\n".join(lines) + "\n"


def _run_algorithm(problem: Any, args: argparse.Namespace) -> Result:
    initial_state = random_initial_state(problem, Random(args.seed))

    if args.algorithm == "hill-climbing":
        return steepest_ascent_hill_climbing(initial_state)

    if args.algorithm == "genetic-algorithm":
        return genetic_algorithm(
            initial_state,
            population_size=args.population_size,
            max_iterations=args.iterations,
            seed=args.seed,
        )

    return simulated_annealing(
        initial_state,
        initial_temperature=args.initial_temperature,
        cooling_rate=args.cooling_rate,
        minimum_temperature=args.minimum_temperature,
        max_iterations=args.iterations,
        seed=args.seed,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a local-search algorithm for a truck-loading problem."
    )
    parser.add_argument(
        "json_file",
        type=Path,
        nargs="?",
        default=DEFAULT_INPUT_FILE,
        help="problem JSON path or filename in test/input (default: test-1)",
    )
    parser.add_argument(
        "--algorithm",
        choices=(
            "hill-climbing",
            "simulated-annealing",
            "genetic-algorithm",
        ),
        default="hill-climbing",
        help="algorithm to run (default: hill-climbing)",
    )
    parser.add_argument(
        "--iterations",
        type=_non_negative_int,
        default=500,
        help="maximum iterations/generations (default: 500)",
    )
    parser.add_argument(
        "--population-size",
        type=_non_negative_int,
        default=50,
        help="population size for genetic algorithm (default: 50)",
    )
    parser.add_argument(
        "--initial-temperature",
        type=float,
        default=100.0,
        help="initial temperature for simulated annealing (default: 100)",
    )
    parser.add_argument(
        "--cooling-rate",
        type=float,
        default=0.995,
        help="cooling rate for simulated annealing (default: 0.995)",
    )
    parser.add_argument(
        "--minimum-temperature",
        type=float,
        default=0.1,
        help="minimum temperature for simulated annealing (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="optional random seed for reproducible local-search runs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="directory for the generated JSON and TXT files",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        input_path = resolve_problem_file(args.json_file, DEFAULT_INPUT_DIRECTORY)
        problem = json_to_problem(input_path)
        result = _run_algorithm(problem, args)
    except (OSError, TypeError, ValueError, KeyError) as err:
        parser.error(str(err))

    output_directory = args.output_dir.expanduser()
    output_directory.mkdir(parents=True, exist_ok=True)
    snapshot_alias = SNAPSHOT_ALIASES[args.algorithm]
    details_path = output_directory / f"{input_path.stem}-{snapshot_alias}.txt"
    snapshot_path = output_directory / (
        f"{input_path.stem}-{snapshot_alias}.json"
    )

    details_path.write_text(
        _format_details(input_path, args.algorithm, result),
        encoding="utf-8",
    )
    write_snapshot(snapshot_path, result, args.algorithm)

    print(f"snapshot JSON: {snapshot_path}")
    print(f"details TXT: {details_path}")
    print(f"final value: {result.state.value}")


if __name__ == "__main__":
    main()
