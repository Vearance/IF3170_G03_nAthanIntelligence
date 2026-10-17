import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.algo.simulated_annealing import simulated_annealing
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
        truck=Truck(dimensions=Dimensions(w=2, l=1, h=1), maxCapacity=2),
    )


class SimulatedAnnealingTests(unittest.TestCase):
    def test_returns_valid_best_state_without_mutating_initial_state(self) -> None:
        initial_state = State.empty(make_problem())

        result = simulated_annealing(
            initial_state,
            initial_temperature=10,
            cooling_rate=0.9,
            minimum_temperature=0.1,
            max_iterations=25,
            seed=3,
        )

        self.assertIsInstance(result, Result)
        self.assertTrue(Evaluate.is_valid(result.initial_state))
        self.assertTrue(Evaluate.is_valid(result.state))
        self.assertEqual(result.initial_state.value, 0)
        self.assertEqual(initial_state.package_ids_in_grid(), set())
        self.assertEqual(len(result.value_history), result.iterations + 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
