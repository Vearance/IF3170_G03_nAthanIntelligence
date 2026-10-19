import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.algo.genetic import genetic_algorithm
from src.models.state import State
from src.utils.evaluate import Evaluate
from src.utils.parser import json_to_problem
from src.utils.plotting import create_genetic_plot
from src.utils.snapshot import snapshot_to_dict

CONTROL_POPULATION = 20
CONTROL_ITERATIONS = 30
ITERATION_VARIATIONS = (10, 30, 50)
POPULATION_VARIATIONS = (10, 20, 30)


# func: menjalankan dua kelompok eksperimen GA sesuai spek
def main() -> None:
    parser = argparse.ArgumentParser(description="Jalankan 18 eksperimen Genetic Algorithm.")
    parser.add_argument("input", type=Path, nargs="?",
                        default=ROOT / "test/input/test-4.json")
    parser.add_argument("--output", type=Path, default=ROOT / "output/ga")
    parser.add_argument("--name", default="results",
                        help="Nama folder dan awalan file hasil (default: results)")
    args = parser.parse_args()
    if args.name in ("", ".", "..") or Path(args.name).name != args.name:
        parser.error("--name harus berupa nama, bukan path direktori")

    problem = json_to_problem(args.input)
    output = args.output / args.input.stem / args.name
    try:
        output.mkdir(parents=True)
    except FileExistsError:
        parser.error(f"Folder {output} sudah ada; gunakan --name lain")
    configurations = [
        ("populasi_tetap", CONTROL_POPULATION, iterations)
        for iterations in ITERATION_VARIATIONS
    ] + [
        ("iterasi_tetap", population, CONTROL_ITERATIONS)
        for population in POPULATION_VARIATIONS
    ]

    with (output / f"{args.name}_summary.csv").open("w", newline="", encoding="utf-8") as file:
        fields = ["group", "run", "seed", "population_size", "iterations",
                  "initial_value", "final_value", "duration_seconds"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for group, population, iterations in configurations:
            for run in range(1, 4):
                seed = run if group == "populasi_tetap" else run + 3
                result = genetic_algorithm(
                    State.empty(problem), population_size=population,
                    max_iterations=iterations, seed=seed,
                )
                if not (Evaluate.is_valid(result.initial_state)
                        and Evaluate.is_valid(result.state)):
                    raise ValueError("Eksperimen menghasilkan state invalid")

                summary = {
                    "group": group,
                    "run": run,
                    "seed": seed,
                    "population_size": population,
                    "iterations": result.iterations,
                    "initial_value": result.initial_state.value,
                    "final_value": result.state.value,
                    "duration_seconds": result.duration_seconds,
                }
                filename = f"{args.name}_{group}_n{population}_g{iterations}_run{run}"
                record = {
                    **snapshot_to_dict(result, "genetic-algorithm"),
                    **summary,
                    "max_value_history": result.value_history,
                    "mean_value_history": result.metrics["mean_value_history"],
                    "metrics": result.metrics,
                }
                (output / f"{filename}.json").write_text(
                    json.dumps(record, indent=2, allow_nan=False), encoding="utf-8"
                )
                title = f"{group.replace('_', ' ')} | N={population}, G={iterations}, run={run}"
                figure = create_genetic_plot(result, title)
                figure.savefig(output / f"{filename}.png")
                figure.clear()
                writer.writerow(summary)
                file.flush()
                print(f"{filename}: value={result.state.value}, "
                      f"durasi={result.duration_seconds:.3f}s", flush=True)

    print(f"Hasil run tersimpan di: {output}", flush=True)


if __name__ == "__main__":
    main()
