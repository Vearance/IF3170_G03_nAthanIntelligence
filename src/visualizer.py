import argparse
import colorsys
import inspect
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import viser

from src.algo.hill_climb import steepest_ascent_hill_climbing
from src.algo.simulated_annealing import simulated_annealing
from src.models.state import Grid, State
from src.models.types.orientation import Orientation
from src.utils.evaluate import Evaluate
from src.utils.parser import json_to_problem

Algorithm = Callable[..., Any]


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
    grid: Grid = [
        [[None for _ in range(truck.w)] for _ in range(truck.l)]
        for _ in range(truck.h)
    ]
    for package_id, (x_start, y_start, z_start) in placements.items():
        package = problem.get_package(package_id)
        orientation = package.orientation
        if orientations is not None and package_id in orientations:
            orientation = orientations[package_id]
        width, length, height = _axis_dimensions(package, orientation)
        for z in range(z_start, z_start + height):
            for y in range(y_start, y_start + length):
                for x in range(x_start, x_start + width):
                    if not (0 <= x < truck.w and 0 <= y < truck.l and 0 <= z < truck.h):
                        continue
                    grid[z][y][x] = package_id
    pkg_orientation = {
        package_id: replace(
            orientations[package_id]
            if orientations is not None and package_id in orientations
            else package.orientation
        )
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


def _state_from_result(result: Any) -> State:
    if isinstance(result, State):
        return result

    state = result.get("state") if isinstance(result, Mapping) else getattr(result, "state", None)
    if isinstance(state, State):
        return state

    raise TypeError(
        "algorithm result must be a State or expose a State through the 'state' attribute"
    )


def _result_details(result: Any) -> str:
    details = []
    if isinstance(result, State):
        details.append(f"value={result.value}")
    elif hasattr(result, "__dataclass_fields__"):
        result_state = getattr(result, "state", None)
        if isinstance(result_state, State):
            details.append(f"value={result_state.value}")
        details.extend(
            f"{name}={getattr(result, name)}"
            for name in result.__dataclass_fields__
            if name not in {"state", "initial_state", "value_history"}
        )
    elif isinstance(result, Mapping):
        details.extend(
            f"{key}={value}"
            for key, value in result.items()
            if key not in {"state", "initial_state", "value_history"}
        )

    return ", ".join(details)


def create_server(
    problem,
    initial_state: State | None = None,
    *,
    port: int = 8080,
    algorithms: Mapping[str, Algorithm] | None = None,
    problem_files: Mapping[str, Path] | None = None,
    selected_problem: str | None = None,
):
    algorithms = dict(algorithms or {})
    problem_files = dict(problem_files or {})
    if not algorithms:
        raise ValueError("at least one algorithm must be registered")
    if problem_files:
        selected_problem = selected_problem or next(iter(problem_files))
        if selected_problem not in problem_files:
            raise ValueError("selected_problem harus ada di problem_files")

    server = viser.ViserServer(port=port)
    truck = problem.truck.dimensions
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
    handles = {}
    colors = {
        package_id: _color(index, len(package_ids))
        for index, package_id in enumerate(package_ids)
    }

    def add_truck_scene():
        return (
            server.scene.add_box(
                "/truck",
                dimensions=(truck.w, truck.l, truck.h),
                position=(truck.w / 2, truck.l / 2, truck.h / 2),
                color=(100, 116, 139),
                opacity=0.08,
                wireframe=True,
            ),
            server.scene.add_grid(
                "/floor",
                width=float(truck.w),
                height=float(truck.l),
                plane="xy",
                cell_size=1.0,
                position=(truck.w / 2, truck.l / 2, 0.0),
            ),
        )

    truck_handle, floor_handle = add_truck_scene()

    problem_selector = None
    if problem_files:
        with server.gui.add_folder("Problem"):
            problem_selector = server.gui.add_dropdown(
                "Input JSON",
                tuple(problem_files),
                initial_value=selected_problem,
            )

    with server.gui.add_folder("Local search"):
        algorithm = server.gui.add_dropdown(
            "Algorithm",
            tuple(algorithms),
        )
        visualize_iterations = server.gui.add_checkbox(
            "Visualize each iteration",
            initial_value=False,
        )
        run_search = server.gui.add_button("Run local search", color="green")
        remove_all = server.gui.add_button("Remove all packages", color="red")
        frame_scene_button = server.gui.add_button("Frame scene")

    status = server.gui.add_markdown("Choose an algorithm and run local search.")
    notification = server.gui.add_markdown("")

    def package_position(package_id: str) -> tuple[float, float, float]:
        package = problem.get_package(package_id)
        width, length, height = _axis_dimensions(package, orientations[package_id])
        if package_id in placements:
            x, y, z = placements[package_id]
            return x + width / 2, y + length / 2, z + height / 2
        index = package_ids.index(package_id)
        return (
            truck.w + 2.0 + (index % 4) * 4.0,
            (index // 4) * 4.0 + 2.0,
            height / 2,
        )

    def frame_camera() -> None:
        minimum = [0.0, 0.0, 0.0]
        maximum = [float(truck.w), float(truck.l), float(truck.h)]
        for package_id in package_ids:
            center = package_position(package_id)
            dimensions = _axis_dimensions(
                problem.get_package(package_id), orientations[package_id]
            )
            for axis in range(3):
                minimum[axis] = min(
                    minimum[axis], center[axis] - dimensions[axis] / 2
                )
                maximum[axis] = max(
                    maximum[axis], center[axis] + dimensions[axis] / 2
                )

        look_at = (
            (minimum[0] + maximum[0]) / 2,
            (minimum[1] + maximum[1]) / 2,
            (minimum[2] + maximum[2]) / 2,
        )
        distance = max(maximum[axis] - minimum[axis] for axis in range(3))
        camera_position = (
            look_at[0] + distance,
            look_at[1] - distance,
            look_at[2] + distance * 0.75,
        )
        server.initial_camera.position = camera_position
        server.initial_camera.look_at = look_at
        server.initial_camera.up = (0.0, 0.0, 1.0)

        for client in server.get_clients().values():
            with client.atomic():
                client.camera.position = camera_position
                client.camera.look_at = look_at

    def refresh() -> None:
        state = _make_state(problem, placements, orientations)
        for package_id, handle in handles.items():
            package = problem.get_package(package_id)
            handle.dimensions = _axis_dimensions(package, orientations[package_id])
            handle.position = package_position(package_id)
            handle.color = colors[package_id]
            handle.opacity = 0.82 if package_id in placements else 0.35

        weight = sum(problem.get_package(package_id).weight for package_id in placements)
        validity = "valid" if Evaluate.is_valid(state) else "invalid"
        status.content = (
            f"algorithm: **{algorithm.value}**  |  "
            f"**{len(placements)} placed**  |  weight: **{weight:g} / "
            f"{problem.truck.maxCapacity:g}**  |  state: **{validity}**"
        )

    def animate_placements(
        rplace: dict[str, tuple[int, int, int]],
        rori: dict[str, Orientation],
    ) -> None:
        placements.clear()
        orientations.clear()
        orientations.update(rori)
        refresh()

        for package_id in sorted(
            rplace,
            key=lambda item: rplace[item][::-1],
        ):
            placements[package_id] = rplace[package_id]
            refresh()
            handles[package_id].opacity = 0.15
            time.sleep(0.05)
            handles[package_id].opacity = 0.80
            time.sleep(0.05)

        refresh()

    def animate_iteration(state: State, _iteration: int) -> None:
        nplace = _placements_from_state(state)
        nori = _orientations_from_state(state)
        changed = {
            package_id
            for package_id in package_ids
            if placements.get(package_id) != nplace.get(package_id)
            or orientations[package_id] != nori[package_id]
        }

        placements.clear()
        placements.update(nplace)
        orientations.clear()
        orientations.update(nori)
        refresh()

        for package_id in changed:
            handles[package_id].opacity = 0.15
        time.sleep(0.005)
        for package_id in changed:
            handles[package_id].opacity = 0.80 if package_id in placements else 0.35
        time.sleep(0.005)

    def add_package_handles() -> None:
        for package_id in package_ids:
            package = problem.get_package(package_id)
            dimensions = _axis_dimensions(package, orientations[package_id])
            handles[package_id] = server.scene.add_box(
                f"/packages/{package_id}",
                dimensions=dimensions,
                position=package_position(package_id),
                color=colors[package_id],
                opacity=0.55,
            )

    def load_problem_file(path: Path) -> None:
        nonlocal colors
        nonlocal floor_handle, handles
        nonlocal orientations, package_ids, placements, problem, truck
        nonlocal truck_handle

        try:
            new_problem = json_to_problem(path)
        except (OSError, TypeError, ValueError, KeyError) as err:
            notification.content = f"**Could not load {path.name}:** {err}"
            return

        for handle in handles.values():
            handle.remove()
        truck_handle.remove()
        floor_handle.remove()

        problem = new_problem
        truck = problem.truck.dimensions
        package_ids = tuple(problem.packages)
        placements = {}
        orientations = {
            package_id: replace(package.orientation)
            for package_id, package in problem.packages.items()
        }
        handles = {}
        colors = {
            package_id: _color(index, len(package_ids))
            for index, package_id in enumerate(package_ids)
        }
        truck_handle, floor_handle = add_truck_scene()
        add_package_handles()
        refresh()
        frame_camera()
        notification.content = f"**Loaded {path.name}.**"

    add_package_handles()

    if problem_selector is not None:
        @problem_selector.on_update
        def _(_) -> None:
            load_problem_file(problem_files[problem_selector.value])

    @frame_scene_button.on_click
    def _(_) -> None:
        frame_camera()

    @run_search.on_click
    def _(_) -> None:
        runner = algorithms[algorithm.value]
        parameters = inspect.signature(runner).parameters
        initial_state = _make_state(problem, placements, orientations)
        if "on_iteration" in parameters and visualize_iterations.value:
            result = runner(initial_state, on_iteration=animate_iteration)
        else:
            result = runner(initial_state)
        rstate = _state_from_result(result)
        if rstate.problem is not problem:
            raise ValueError("algorithm result belongs to a different problem")
        if not Evaluate.is_valid(rstate):
            notification.content = (
                f"**{algorithm.value} failed:** the search returned an invalid placement."
            )
            return
        # Replay final placements one by one; keep disabled for now.
        # animate_placements(
        #     _placements_from_state(rstate),
        #     _orientations_from_state(rstate),
        # )
        placements.clear()
        placements.update(_placements_from_state(rstate))
        orientations.clear()
        orientations.update(_orientations_from_state(rstate))
        refresh()
        details = _result_details(result)
        suffix = f": {details}" if details else ""
        notification.content = f"**Placement completed** - {algorithm.value}{suffix}."

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
        notification.content = "**All packages removed.** The truck is in a valid empty state."
        refresh()

    algorithm.on_update(lambda _: refresh())

    refresh()
    frame_camera()
    return server


def _resolve_problem_file(json_file: Path, input_directory: Path) -> Path:
    json_file = json_file.expanduser()
    if json_file.is_file():
        return json_file.resolve()

    if json_file.parent == Path("."):
        file_name = (
            json_file.name
            if json_file.suffix.lower() == ".json"
            else f"{json_file.name}.json"
        )
        input_path = input_directory / file_name
        if input_path.is_file():
            return input_path.resolve()

    raise FileNotFoundError(f"problem JSON tidak ditemukan: {json_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize packages being loaded into a truck."
    )
    parser.add_argument(
        "json_file",
        type=Path,
        nargs="?",
        default=Path("test/input/test-1.json"),
        help="filename in test/input",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="run simulated annealing before opening the visualizer",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=500,
        help="maximum iterations for Simulated Annealing",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    input_directory = Path(__file__).resolve().parents[1] / "test" / "input"
    problem_files = {
        path.name: path.resolve()
        for path in sorted(input_directory.glob("*.json"))
    }
    selected_path = _resolve_problem_file(args.json_file, input_directory)
    selected_problem = next(
        (
            name
            for name, path in problem_files.items()
            if path == selected_path
        ),
        selected_path.name,
    )
    if (
        selected_problem in problem_files
        and problem_files[selected_problem] != selected_path
    ):
        selected_problem = str(selected_path)
    problem_files[selected_problem] = selected_path

    problem = json_to_problem(selected_path)
    state = State.empty(problem)
    if args.simulate:
        state = simulated_annealing(
            state,
            max_iterations=args.iterations,
            seed=args.seed,
        ).state

    create_server(
        problem,
        state,
        port=args.port,
        problem_files=problem_files,
        selected_problem=selected_problem,
        algorithms={
            "Steepest-Ascent Hill-Climbing": lambda current_state, on_iteration=None: (
                steepest_ascent_hill_climbing(
                    current_state,
                    on_iteration=on_iteration,
                )
            ),
            "Simulated Annealing": lambda current_state, on_iteration=None: (
                simulated_annealing(
                    current_state,
                    max_iterations=args.iterations,
                    seed=args.seed,
                    on_iteration=on_iteration,
                )
            ),
        },
    )
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()