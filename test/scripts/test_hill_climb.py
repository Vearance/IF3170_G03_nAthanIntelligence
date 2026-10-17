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
from src.utils.evaluate import Evaluate


def make_problem() -> Problem:
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
        for package_id, value in (("P1", 6), ("P2", 10))
    }
    return Problem(
        packages=packages,
        truck=Truck(dimensions=dimensions, maxCapacity=10),
    )


class HillClimbTests(unittest.TestCase):
    def test_steepest_ascent_selects_highest_value_neighbor(self) -> None:
        initial_state = State.empty(make_problem())

        result = steepest_ascent_hill_climbing(initial_state)

        self.assertIsInstance(result, Result)
        self.assertEqual(result.initial_state.value, 0)
        self.assertEqual(result.state.value, 10)
        self.assertEqual(result.value_history, [0, 10, 10])
        self.assertEqual(result.iterations, 2)
        self.assertEqual(result.metrics["termination_reason"], "local_optimum")
        self.assertEqual(result.state.package_ids_in_grid(), {"P2"})
        self.assertTrue(Evaluate.is_valid(result.state))
        self.assertEqual(initial_state.package_ids_in_grid(), set())

    def test_max_iterations_stops_after_requested_neighborhood_scans(self) -> None:
        initial_state = State.empty(make_problem())
        reported_iterations: list[int] = []

        result = steepest_ascent_hill_climbing(
            initial_state,
            max_iterations=1,
            on_iteration=lambda _state, iteration: reported_iterations.append(
                iteration
            ),
        )

        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.value_history, [0, 10])
        self.assertEqual(result.metrics["termination_reason"], "max_iterations")
        self.assertEqual(reported_iterations, [1])
        self.assertTrue(Evaluate.is_valid(result.state))

    def test_negative_iteration_limit_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            steepest_ascent_hill_climbing(
                State.empty(make_problem()), max_iterations=-1
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
