import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.algo.simulated_annealing import simulated_annealing
from src.models.package import Package
from src.models.problem import Problem
from src.models.result import Result
from src.models.state import State
from src.models.truck import Truck
from src.models.types.dims import Dimensions
from src.models.types.orientation import Orientation
from src.models.types.position import Position
from src.utils.evaluate import Evaluate
from src.utils.move import try_place, try_rotate
from src.utils.plotting import create_annealing_plot


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
    def test_history_tracks_current_while_result_keeps_best(self) -> None:
        empty = State.empty(make_problem())
        initial = try_place(empty, "P1", Position(0, 0, 0))
        assert initial is not None
        observed: list[int] = []

        with patch(
            "src.algo.simulated_annealing._random_neighbour", return_value=empty
        ), patch("src.algo.simulated_annealing.random.Random.random", return_value=0):
            result = simulated_annealing(
                initial,
                max_iterations=1,
                on_iteration=lambda state, iteration: observed.append(state.value),
            )

        self.assertEqual(result.value_history, [6, 0])
        self.assertEqual(result.value_history[1:], observed)
        self.assertEqual(result.state.value, 6)
        self.assertEqual(result.metrics["stuck_frequency"], 0)

    def test_records_actual_probability_and_missing_candidates(self) -> None:
        empty = State.empty(make_problem())
        initial = try_place(empty, "P1", Position(0, 0, 0))
        assert initial is not None

        with patch(
            "src.algo.simulated_annealing._random_neighbour",
            side_effect=[empty, initial, None],
        ):
            result = simulated_annealing(
                initial,
                initial_temperature=6 / math.log(2),
                max_iterations=3,
                seed=0,
            )

        history = result.metrics["acceptance_probability_history"]
        assert isinstance(history, list)
        self.assertAlmostEqual(history[0], 0.5)
        self.assertEqual(history[1:], [1.0, None])
        self.assertEqual(result.metrics["accepted_moves"], 1)
        self.assertEqual(result.metrics["stuck_frequency"], 3)

        figure = create_annealing_plot(result)
        line = figure.axes[0].lines[0]
        self.assertEqual(list(line.get_xdata()), [1, 2, 3])
        self.assertTrue(math.isnan(line.get_ydata()[2]))
        figure.clear()

    def test_counts_unchanged_states_instead_of_unchanged_values(self) -> None:
        initial = State.empty(make_problem())
        improved = try_place(initial, "P1", Position(0, 0, 0))
        assert improved is not None
        rotated = try_rotate(improved, "P2", "z")
        assert rotated is not None

        with patch(
            "src.algo.simulated_annealing._random_neighbour",
            side_effect=[initial.copy(), None, improved, rotated, initial, initial.copy()],
        ), patch("src.algo.simulated_annealing.random.Random.random", return_value=0):
            result = simulated_annealing(initial, max_iterations=6)

        self.assertEqual(result.metrics["stuck_frequency"], 3)
        self.assertEqual(result.value_history, [0, 0, 0, 6, 6, 0, 0])

    def test_zero_iterations_has_no_probabilities_or_stuck_count(self) -> None:
        result = simulated_annealing(State.empty(make_problem()), max_iterations=0)
        self.assertEqual(result.metrics["acceptance_probability_history"], [])
        self.assertEqual(result.metrics["stuck_frequency"], 0)
        figure = create_annealing_plot(result)
        self.assertEqual(len(figure.axes[0].lines[0].get_xdata()), 0)
        figure.clear()

    def test_failed_sampling_still_cools_and_records_iterations(self) -> None:
        initial_state = State.empty(make_problem())
        callbacks: list[int] = []

        with patch(
            "src.algo.simulated_annealing._random_neighbour", return_value=None
        ):
            result = simulated_annealing(
                initial_state,
                initial_temperature=8,
                cooling_rate=0.5,
                minimum_temperature=1,
                max_iterations=10,
                on_iteration=lambda state, iteration: callbacks.append(iteration),
            )

        self.assertEqual(result.iterations, 3)
        self.assertEqual(result.value_history, [0, 0, 0, 0])
        self.assertEqual(callbacks, [1, 2, 3])
        self.assertEqual(result.metrics["accepted_moves"], 0)
        self.assertEqual(result.metrics["acceptance_probability_history"], [None] * 3)
        self.assertEqual(result.metrics["stuck_frequency"], 3)
        self.assertEqual(result.state, initial_state)

    def test_rotates_outside_package_before_placing_it(self) -> None:
        problem = make_problem()
        problem.packages = {"P1": problem.packages["P1"]}
        problem.packages["P1"].dimensions = Dimensions(w=2, l=1, h=1)
        problem.truck.dimensions = Dimensions(w=1, l=2, h=1)
        initial_state = State.empty(problem)

        result = simulated_annealing(
            initial_state, max_iterations=100, seed=0
        )

        self.assertEqual(result.state.value, 6)
        self.assertTrue(Evaluate.is_valid(result.state))
        self.assertEqual(initial_state.out_pkg, {"P1"})
        self.assertEqual(initial_state.pkg_orientation["P1"], Orientation("x", "y", "z"))

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
