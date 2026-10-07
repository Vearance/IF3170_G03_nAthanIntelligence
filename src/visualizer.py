import argparse
import colorsys
import time
from pathlib import Path

import viser

from src.models.state import State
from src.utils.evaluate import Evaluate
from src.utils.parser import json_to_problem


def _color(index: int, total: int) -> tuple[int, int, int]:
    hue = index / max(total, 1)
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
    return round(red * 255), round(green * 255), round(blue * 255)


def _axis_dimensions(package) -> tuple[int, int, int]:
    dimensions = {
        package.orientation.w: package.dimensions.w,
        package.orientation.l: package.dimensions.l,
        package.orientation.h: package.dimensions.h,
    }
    return dimensions["x"], dimensions["y"], dimensions["z"]


def _make_state(problem, placements: dict[str, tuple[int, int, int]]) -> State:
    truck = problem.truck.dimensions
    extents = [truck.w, truck.l, truck.h]
    for package_id, (x, y, z) in placements.items():
        package = problem.get_package(package_id)
        width, length, height = _axis_dimensions(package)
        extents[0] = max(extents[0], x + width)
        extents[1] = max(extents[1], y + length)
        extents[2] = max(extents[2], z + height)

    grid = [
        [[None for _ in range(extents[0])] for _ in range(extents[1])]
        for _ in range(extents[2])
    ]
    for package_id, (x_start, y_start, z_start) in placements.items():
        package = problem.get_package(package_id)
        width, length, height = _axis_dimensions(package)
        for z in range(z_start, z_start + height):
            for y in range(y_start, y_start + length):
                for x in range(x_start, x_start + width):
                    if grid[z][y][x] not in (None, package_id):
                        continue
                    grid[z][y][x] = package_id
    return State(problem, grid, {}, set(), 0)


def create_server(problem):
    server = viser.ViserServer()
    truck_dimensions = problem.truck.dimensions
    package_ids = tuple(problem.packages)
    placements: dict[str, tuple[int, int, int]] = {}
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
        )

    with server.gui.add_folder("Package input"):
        selected = server.gui.add_dropdown("Package", package_ids)
        position_x = server.gui.add_slider("X", 0, max(truck_dimensions.w - 1, 0), 1, 0)
        position_y = server.gui.add_slider("Y", 0, max(truck_dimensions.l - 1, 0), 1, 0)
        position_z = server.gui.add_slider("Z", 0, max(truck_dimensions.h - 1, 0), 1, 0)
        place = server.gui.add_button("Place selected")
        clear = server.gui.add_button("Clear selected")
        status = server.gui.add_markdown("Select a package to begin.")

    def package_position(package_id: str) -> tuple[float, float, float]:
        package = problem.get_package(package_id)
        width, length, height = _axis_dimensions(package)
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
        state = _make_state(problem, placements)
        selected_id = selected.value
        for package_id, handle in package_handles.items():
            handle.position = package_position(package_id)
            handle.color = colors[package_id]
            handle.opacity = 0.82 if package_id == selected_id else 0.55

        weight = sum(problem.get_package(package_id).weight for package_id in placements)
        validity = "valid" if Evaluate.is_valid(state) else "invalid"
        status.content = (
            f"algorithm: **{algorithm.value}**  |  "
            f"variant: **{hill_climbing_variant.value}**  |  "
            f"**{len(placements)} placed**  |  weight: **{weight:g} / "
            f"{problem.truck.maxCapacity:g}**  |  state: **{validity}**"
        )

    for package_id in package_ids:
        package = problem.get_package(package_id)
        dimensions = _axis_dimensions(package)
        package_handles[package_id] = server.scene.add_box(
            f"/packages/{package_id}",
            dimensions=dimensions,
            position=package_position(package_id),
            color=colors[package_id],
            opacity=0.55,
        )

    def sync_sliders(_) -> None:
        package_id = selected.value
        if package_id in placements:
            position_x.value, position_y.value, position_z.value = placements[package_id]
        refresh()

    @selected.on_update
    def _(_) -> None:
        sync_sliders(_)

    @place.on_click
    def _(_) -> None:
        placements[selected.value] = (position_x.value, position_y.value, position_z.value)
        refresh()

    @clear.on_click
    def _(_) -> None:
        placements.pop(selected.value, None)
        refresh()

    for control in (position_x, position_y, position_z):
        control.on_update(lambda _: refresh())
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
    args = parser.parse_args()
    create_server(json_to_problem(args.json_file))
    print("Viser server running at http://localhost:8080")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()