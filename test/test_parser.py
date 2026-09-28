import argparse
from pathlib import Path

from src.utils.parser import json_to_problem, problem_to_json


def main() -> None:
    argument_parser = argparse.ArgumentParser(
        description="Parse a problem JSON file into domain objects."
    )
    argument_parser.add_argument(
        "json_file",
        type=Path,
        help="path to the problem JSON file",
    )
    args = argument_parser.parse_args()

    problem = json_to_problem(args.json_file)

    print("=== JSON -> OBJECT ===")
    print(f"problem.truck = {problem.truck}")
    print(f"problem.truck.dimensions = {problem.truck.dimensions}")
    print(f"problem.truck.dimensions.w = {problem.truck.dimensions.w}")
    print(f"problem.truck.dimensions.l = {problem.truck.dimensions.l}")
    print(f"problem.truck.dimensions.h = {problem.truck.dimensions.h}")
    print(f"problem.truck.maxCapacity = {problem.truck.maxCapacity}")
    print(f"problem.packages = {problem.packages}")

    for package_id, package in problem.packages.items():
        package_name = f"problem.packages[{package_id!r}]"
        print(f"{package_name}.dimensions = {package.dimensions}")
        print(f"{package_name}.dimensions.w = {package.dimensions.w}")
        print(f"{package_name}.dimensions.l = {package.dimensions.l}")
        print(f"{package_name}.dimensions.h = {package.dimensions.h}")
        print(f"{package_name}.orientation = {package.orientation}")
        print(f"{package_name}.orientation.w = {package.orientation.w}")
        print(f"{package_name}.orientation.l = {package.orientation.l}")
        print(f"{package_name}.orientation.h = {package.orientation.h}")
        print(f"{package_name}.value = {package.value}")
        print(f"{package_name}.weight = {package.weight}")
        print(f"{package_name}.isFragile = {package.isFragile}")
        print(f"{package_name}.eta = {package.eta}")

    print("\n=== OBJECT -> JSON ===")
    print(problem_to_json(problem))


if __name__ == "__main__":
    main()
