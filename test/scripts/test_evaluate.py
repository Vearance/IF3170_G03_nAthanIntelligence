import unittest
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.package import Package
from src.models.problem import Problem
from src.models.state import State
from src.models.truck import Truck
from src.models.types.dims import Dimensions
from src.models.types.orientation import Orientation
from src.utils.evaluate import Evaluate
from src.utils.parser import json_to_problem


ORIENTATION = Orientation(w="x", l="y", h="z")


def make_package(
    package_id: str,
    *,
    weight: float = 5,
    value: int = 10,
    fragile: bool = False,
) -> Package:
    return Package(
        id=package_id,
        dimensions=Dimensions(w=2, l=2, h=1),
        orientation=ORIENTATION,
        value=value,
        weight=weight,
        isFragile=fragile,
        eta=1,
    )


def make_state(
    packages: list[Package],
    placements: dict[str, tuple[int, int, int]],
    *,
    capacity: int = 50,
    truck_dimensions: Dimensions = Dimensions(w=4, l=4, h=4),
) -> State:
    problem = Problem(
        packages={package.id: package for package in packages},
        truck=Truck(dimensions=truck_dimensions, maxCapacity=capacity),
    )
    grid = [
        [[None for _ in range(truck_dimensions.w)] for _ in range(truck_dimensions.l)]
        for _ in range(truck_dimensions.h)
    ]
    for package_id, (x_start, y_start, z_start) in placements.items():
        package = problem.get_package(package_id)
        for z in range(z_start, z_start + package.dimensions.h):
            for y in range(y_start, y_start + package.dimensions.l):
                for x in range(x_start, x_start + package.dimensions.w):
                    if z < len(grid) and y < len(grid[z]) and x < len(grid[z][y]):
                        grid[z][y][x] = package_id
                    else:
                        # Keep the out-of-bounds placement visible to the evaluator.
                        while len(grid) <= z:
                            grid.append([])
                        while len(grid[z]) <= y:
                            grid[z].append([])
                        while len(grid[z][y]) <= x:
                            grid[z][y].append(None)
                        grid[z][y][x] = package_id
    return State(problem, grid, {}, set(), 0)


def make_parsed_state(problem, package_id: str) -> State:
    truck_dimensions = problem.truck.dimensions
    package = problem.get_package(package_id)
    dimensions_by_axis = {
        axis: getattr(package.dimensions, dimension)
        for dimension, axis in (
            ("w", package.orientation.w),
            ("l", package.orientation.l),
            ("h", package.orientation.h),
        )
    }
    grid = [
        [[None for _ in range(truck_dimensions.w)] for _ in range(truck_dimensions.l)]
        for _ in range(truck_dimensions.h)
    ]
    for z in range(dimensions_by_axis["z"]):
        for y in range(dimensions_by_axis["y"]):
            for x in range(dimensions_by_axis["x"]):
                grid[z][y][x] = package_id
    return State(problem, grid, {}, set(), 0)


class EvaluateTests(unittest.TestCase):
    def test_parser_json_can_be_evaluated(self) -> None:
        for input_name in ("test-1.json", "test-2.json"):
            with self.subTest(input_name=input_name):
                problem = json_to_problem(
                    Path(__file__).resolve().parents[1] / "input" / input_name
                )
                package_id = next(iter(problem.packages))
                state = make_parsed_state(problem, package_id)

                self.assertTrue(problem.packages)
                self.assertTrue(Evaluate.inside_truck(state))
                self.assertTrue(Evaluate.no_floating(state))
                self.assertTrue(Evaluate.within_capacity(state))
                self.assertTrue(Evaluate.is_valid(state))
                self.assertEqual(
                    Evaluate.value(state), problem.packages[package_id].value
                )

    def test_value_counts_packages_inside_truck(self) -> None:
        package = make_package("P1", value=42)
        state = make_state([package], {"P1": (0, 0, 0)})

        self.assertEqual(Evaluate.value(state), 42)

    def test_package_on_floor_is_valid(self) -> None:
        package = make_package("P1")
        state = make_state([package], {"P1": (0, 0, 0)})

        self.assertTrue(Evaluate.inside_truck(state))
        self.assertTrue(Evaluate.no_floating(state))
        self.assertTrue(Evaluate.is_valid(state))

    def test_package_without_support_is_invalid(self) -> None:
        package = make_package("P1")
        state = make_state([package], {"P1": (0, 0, 1)})

        self.assertFalse(Evaluate.no_floating(state))
        self.assertFalse(Evaluate.is_valid(state))

    def test_package_cannot_penetrate_truck_wall(self) -> None:
        package = make_package("P1")
        state = make_state([package], {"P1": (3, 0, 0)})

        self.assertFalse(Evaluate.inside_truck(state))
        self.assertFalse(Evaluate.is_valid(state))

    def test_fragile_package_cannot_support_another_package(self) -> None:
        fragile = make_package("P1", fragile=True)
        upper = make_package("P2")
        state = make_state(
            [fragile, upper],
            {"P1": (0, 0, 0), "P2": (0, 0, 1)},
        )

        self.assertTrue(Evaluate.no_floating(state))
        self.assertFalse(Evaluate.fragile_not_supporting(state))
        self.assertFalse(Evaluate.is_valid(state))

    def test_total_weight_cannot_exceed_capacity(self) -> None:
        package = make_package("P1", weight=6)
        state = make_state([package], {"P1": (0, 0, 0)}, capacity=5)

        self.assertFalse(Evaluate.within_capacity(state))
        self.assertFalse(Evaluate.is_valid(state))


if __name__ == "__main__":
    unittest.main(verbosity=2)