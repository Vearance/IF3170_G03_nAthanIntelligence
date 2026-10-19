import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.algo.genetic import genetic_algorithm
from src.models.package import Package
from src.models.problem import Problem
from src.models.result import Result
from src.models.state import State
from src.models.truck import Truck
from src.models.types.dims import Dimensions
from src.models.types.orientation import Orientation
from src.utils.evaluate import Evaluate


def make_problem() -> Problem:
    packages = {
        package_id: Package(
            id=package_id,
            dimensions=Dimensions(w=1, l=1, h=1),
            orientation=Orientation(w="x", l="y", h="z"),
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


class GeneticTests(unittest.TestCase):
    def test_returns_valid_result_without_mutating_initial_state(self) -> None:
        initial_state = State.empty(make_problem())
        reported_iterations: list[int] = []

        result = genetic_algorithm(
            initial_state,
            population_size=5,
            max_iterations=10,
            seed=3,
            on_iteration=lambda _state, iteration: reported_iterations.append(iteration),
        )

        self.assertIsInstance(result, Result)
        self.assertTrue(Evaluate.is_valid(result.initial_state))
        self.assertTrue(Evaluate.is_valid(result.state))
        self.assertEqual(initial_state, State.empty(initial_state.problem))
        self.assertEqual(result.iterations, 10)
        self.assertEqual(result.metrics["population_size"], 5)
        self.assertEqual(len(result.value_history), result.iterations + 1)
        mean_history = result.metrics["mean_value_history"]
        if not isinstance(mean_history, list):
            self.fail("mean_value_history harus berupa list")
        self.assertEqual(len(mean_history), result.iterations + 1)
        self.assertEqual(result.value_history, sorted(result.value_history))
        self.assertEqual(result.state.value, Evaluate.value(result.state))
        self.assertEqual(result.state.value, result.value_history[-1])
        self.assertEqual(reported_iterations, list(range(1, 11)))
        self.assertGreaterEqual(result.duration_seconds, 0)

    def test_same_seed_produces_same_states_and_history(self) -> None:
        initial_state = State.empty(make_problem())

        first = genetic_algorithm(
            initial_state, population_size=5, max_iterations=5, seed=12
        )
        second = genetic_algorithm(
            initial_state, population_size=5, max_iterations=5, seed=12
        )

        self.assertEqual(first.initial_state, second.initial_state)
        self.assertEqual(first.state, second.state)
        self.assertEqual(first.value_history, second.value_history)
        self.assertEqual(first.metrics, second.metrics)

    def test_invalid_population_and_iterations_are_rejected(self) -> None:
        initial_state = State.empty(make_problem())
        for parameter, value, error in (
            ("population_size", 1, ValueError),
            ("population_size", True, TypeError),
            ("population_size", 2.5, TypeError),
            ("max_iterations", -1, ValueError),
            ("max_iterations", False, TypeError),
            ("max_iterations", 1.5, TypeError),
        ):
            try:
                # sengaja kirim tipe invalid untuk menguji validasi runtime
                genetic_algorithm(initial_state, **{parameter: value})  # pyright: ignore[reportArgumentType]
            except error:
                continue
            except Exception as actual:
                print(f"gagal: {parameter}={value!r}, "
                      f"expected {error.__name__}, mendapat {type(actual).__name__}")
                raise

            print(f"gagal: {parameter}={value!r} seharusnya ditolak")
            self.fail("parameter invalid tidak ditolak")


if __name__ == "__main__":
    unittest.main(verbosity=2)
