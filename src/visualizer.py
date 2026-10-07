import argparse
import colorsys
import time
from dataclasses import replace
from pathlib import Path

import viser

from src.algo.simulated_annealing import simulated_annealing

from src.models.state import State
from src.models.types.orientation import Orientation
from src.utils.evaluate import Evaluate
from src.utils.parser import json_to_problem


def _color(index: int, total: int) -> tuple[int, int, int]:
    hue = index / max(total, 1)
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
    return round(red * 255), round(green * 255), round(blue * 255)


def _axis_dimensions(package, orientation: Orientation | None = None) -> tuple[int, int, int]:
    orientation = orientation or package.orientation
    dimensions = {
        orientation.w: package.dimensions.w,
        orientation.l: package.dimensions.l,
        orientation.h: package.dimensions.h,
    }
    return dimensions["x"], dimensions["y"], dimensions["z"]


def _make_state(
    problem,
    placements: dict[str, tuple[int, int, int]],
    orientations: dict[str, Orientation] | None = None,
) -> State:
    truck = problem.truck.dimensions
    grid = [
        [[None for _ in range(truck.w)] for _ in range(truck.l)]
        for _ in range(truck.h)
    ]
    for package_id, (x_start, y_start, z_start) in placements.items():
        package = problem.get_package(package_id)
        width, length, height = _axis_dimensions(
            package, (orientations or {}).get(package_id)
        )
        for z in range(z_start, z_start + height):
            for y in range(y_start, y_start + length):
                for x in range(x_start, x_start + width):
                    if not (0 <= x < truck.w and 0 <= y < truck.l and 0 <= z < truck.h):
                        continue
                    grid[z][y][x] = package_id
    pkg_orientation = {
        package_id: replace((orientations or {}).get(package_id, package.orientation))
        for package_id, package in problem.packages.items()
    }
    return State(
        problem,
        grid,
        pkg_orientation,
        set(problem.packages) - set(placements),
        0,
    )


def _placements_from_state(state: State) -> dict[str, tuple[int, int, int]]:
    placements: dict[str, tuple[int, int, int]] = {}
    for package_id in state.package_ids_in_grid():
        cells = [
            (x, y, z)
            for z, layer in enumerate(state.grid)
            for y, row in enumerate(layer)
            for x, cell in enumerate(row)
            if cell == package_id
        ]
        placements[package_id] = (
            min(cell[0] for cell in cells),
            min(cell[1] for cell in cells),
            min(cell[2] for cell in cells),
        )
    return placements


def _orientations_from_state(state: State) -> dict[str, Orientation]:
    return {
        package_id: replace(orientation)
        for package_id, orientation in state.pkg_orientation.items()
    }


def create_server(
    problem,
    initial_state: State | None = None,
    *,
    port: int = 8080,
):
    server = viser.ViserServer(port=port)
    truck_dimensions = problem.truck.dimensions
    package_ids = tuple(problem.packages)
    placements = _placements_from_state(initial_state) if initial_state else {}
    orientations = (
        _orientations_from_state(initial_state)
        if initial_state
        else {
            package_id: replace(package.orientation)
            for package_id, package in problem.packages.items()
        }
    )
    package_handles = {}
    colors = {
        package_id: _color(index, len(package_ids))
        for index, package_id in enumerate(package_ids)
    }

    server.scene.add_box(
        "/truck",
        dimensions=(truck_dimensions.w, truck_dimensions.l, truck_dimensions.h),
        position=(truck_dimensions.w / 2, truck_dimensions.l / 2, truck_dimensions.h / 2),
        color=(100, 116, 139),
        opacity=0.08,
        wireframe=True,
    )
    server.scene.add_grid(
        "/floor",
        width=float(truck_dimensions.w),
        height=float(truck_dimensions.l),
        plane="xy",
        cell_size=1.0,
        position=(truck_dimensions.w / 2, truck_dimensions.l / 2, 0.0),
    )

    with server.gui.add_folder("Local search"):
        algorithm = server.gui.add_dropdown(
            "Algorithm",
            (
                "Choose algorithm...",
                "Hill Climbing",
                "Simulated Annealing",
                "Genetic Algorithm",
            ),
        )
        hill_climbing_variant = server.gui.add_dropdown(
            "Hill climbing variant",
            (
                "Steepest Ascent",
                "Sideways Move",
                "Random Restart",
            ),
            visible=False,
        )
        run_search = server.gui.add_button("Run local search", color="green")
        remove_all = server.gui.add_button("Remove all packages", color="red")

    status = server.gui.add_markdown("Choose an algorithm and run local search.")

    def package_position(package_id: str) -> tuple[float, float, float]:
        package = problem.get_package(package_id)
        width, length, height = _axis_dimensions(package, orientations[package_id])
        if package_id in placements:
            x, y, z = placements[package_id]
            return x + width / 2, y + length / 2, z + height / 2
        index = package_ids.index(package_id)
        return (
            truck_dimensions.w + 2.0 + (index % 4) * 4.0,
            (index // 4) * 4.0 + 2.0,
            height / 2,
        )

    def refresh() -> None:
        state = _make_state(problem, placements, orientations)
        for package_id, handle in package_handles.items():
            package = problem.get_package(package_id)
            handle.dimensions = _axis_dimensions(package, orientations[package_id])
            handle.position = package_position(package_id)
            handle.color = colors[package_id]
            handle.opacity = 0.82 if package_id in placements else 0.35

        weight = sum(problem.get_package(package_id).weight for package_id in placements)
        validity = "valid" if Evaluate.is_valid(state) else "invalid"
        hill_climbing_variant.visible = algorithm.value == "Hill Climbing"
        variant_text = (
            f"variant: **{hill_climbing_variant.value}**  "
            if algorithm.value == "Hill Climbing"
            else ""
        )
        status.content = (
            f"algorithm: **{algorithm.value}**  |  "
            f"{variant_text}| "
            f"**{len(placements)} placed**  |  weight: **{weight:g} / "
            f"{problem.truck.maxCapacity:g}**  |  state: **{validity}**"
        )

    for package_id in package_ids:
        package = problem.get_package(package_id)
        dimensions = _axis_dimensions(package, orientations[package_id])
        package_handles[package_id] = server.scene.add_box(
            f"/packages/{package_id}",
            dimensions=dimensions,
            position=package_position(package_id),
            color=colors[package_id],
            opacity=0.55,
        )

    @run_search.on_click
    def _(_) -> None:
        if algorithm.value != "Simulated Annealing":
            status.content = (
                f"**{algorithm.value}** is not implemented yet. "
                "Choose **Simulated Annealing** to run local search."
            )
            return
        result = simulated_annealing(
            State.empty(problem),
            max_iterations=500,
            seed=0,
        )
        if not Evaluate.is_valid(result.state):
            status.content = "Search returned an invalid state; layout was not changed."
            return
        placements.clear()
        placements.update(_placements_from_state(result.state))
        orientations.clear()
        orientations.update(_orientations_from_state(result.state))
        status.content = (
            f"Simulated annealing completed: **value={result.state.value}**, "
            f"**iterations={result.iterations}**, "
            f"**{len(placements)} packages placed**."
        )
        refresh()

    @remove_all.on_click
    def _(_) -> None:
        placements.clear()
        orientations.clear()
        orientations.update(
            {
                package_id: replace(package.orientation)
                for package_id, package in problem.packages.items()
            }
        )
        status.content = "All packages removed. The truck is in a valid empty state."
        refresh()

    algorithm.on_update(lambda _: refresh())
    hill_climbing_variant.on_update(lambda _: refresh())

    refresh()
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize packages being loaded into a truck.")
    parser.add_argument(
        "json_file",
        type=Path,
        nargs="?",
        default=Path("test/input/test-1.json"),
        help="problem JSON file (default: test/input/test-1.json)",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="run simulated annealing before opening the visualizer",
    )
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    problem = json_to_problem(args.json_file)
    state = State.empty(problem)
    if args.simulate:
        result = simulated_annealing(
            state,
            max_iterations=args.iterations,
            seed=args.seed,
        )
        state = result.state
        print(
            f"Simulated annealing: value={state.value}, "
            f"iterations={result.iterations}, "
            f"accepted_moves={result.accepted_moves}"
        )
    create_server(problem, state, port=args.port)
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()