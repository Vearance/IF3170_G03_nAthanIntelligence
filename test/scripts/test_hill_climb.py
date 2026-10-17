import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.algo.hill_climb import steepest_ascent_hill_climbing
from src.models.package import Package
from src.models.problem import Problem
from src.models.result import Result
from src.models.state import State
from src.models.truck import Truck
from src.models.types.dims import Dimensions
from src.models.types.orientation import Orientation
from src.models.types.position import Position
from src.utils.evaluate import Evaluate
from src.utils.move import try_place


def make_problem(
    values: tuple[tuple[str, int], ...] = (("P1", 6), ("P2", 10)),
) -> Problem:
    dimensions = Dimensions(w=1, l=1, h=1)
    orientation = Orientation(w="x", l="y", h="z")
    packages = {
        package_id: Package(
            id=package_id,
            dimensions=dimensions,
            orientation=orientation,
            value=value,
            weight=1,
            isFragile=False,
            eta=1,
        )
        for package_id, value in values
    }
    return Problem(
        packages=packages,
        truck=Truck(dimensions=dimensions, maxCapacity=10),
    )


class HillClimbTests(unittest.TestCase):
    def test_steepest_ascent_selects_highest_value_neighbor(self) -> None:
        initial_state = State.empty(make_problem())
        reported_iterations: list[int] = []

        result = steepest_ascent_hill_climbing(
            initial_state,
            on_iteration=lambda _state, iteration: reported_iterations.append(
                iteration
            ),
        )

        self.assertIsInstance(result, Result)
        self.assertEqual(result.initial_state.value, 0)
        self.assertEqual(result.state.value, 10)
        self.assertEqual(result.value_history, [0, 10, 10])
        self.assertEqual(result.iterations, 2)
        self.assertEqual(reported_iterations, [1, 2])
        self.assertEqual(result.metrics["termination_reason"], "local_optimum")
        self.assertEqual(result.state.package_ids_in_grid(), {"P2"})
        self.assertTrue(Evaluate.is_valid(result.state))
        self.assertEqual(initial_state.package_ids_in_grid(), set())

    def test_equal_value_tie_keeps_first_package_in_neighbor_order(self) -> None:
        initial_state = State.empty(make_problem((("P1", 10), ("P2", 10))))

        result = steepest_ascent_hill_climbing(initial_state)

        self.assertEqual(result.state.package_ids_in_grid(), {"P1"})

    def test_higher_value_out_package_can_replace_placed_package(self) -> None:
        initial_state = try_place(
            State.empty(make_problem()),
            "P1",
            Position(x=0, y=0, z=0),
        )
        if initial_state is None:
            self.fail("expected P1 to be placeable")

        result = steepest_ascent_hill_climbing(initial_state)

        self.assertEqual(result.state.value, 10)
        self.assertEqual(result.state.package_ids_in_grid(), {"P2"})
        self.assertTrue(Evaluate.is_valid(result.state))


if __name__ == "__main__":
    unittest.main(verbosity=2)
