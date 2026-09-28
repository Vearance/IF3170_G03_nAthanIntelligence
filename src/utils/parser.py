import json
from pathlib import Path
from typing import Any

from src.models.package import Package
from src.models.problem import Problem
from src.models.truck import Truck
from src.models.types.dims import Dimensions
from src.models.types.orientation import Orientation

def json_to_problem(path: str | Path) -> Problem:
    with Path(path).open(encoding="utf-8") as input_file:
        data = json.load(input_file)

    return problem_from_dict(data)


def load_problem(path: str | Path) -> Problem:
    return json_to_problem(path)


def problem_to_dict(problem: Problem) -> dict[str, Any]:
    if not isinstance(problem, Problem):
        raise TypeError("problem harus berupa instance Problem")

    return {
        "truck": {
            "dimensions": _dimensions_to_dict(problem.truck.dimensions),
            "maxCapacity": problem.truck.maxCapacity,
        },
        "packages": [
            {
                "id": package.id,
                "dimensions": _dimensions_to_dict(package.dimensions),
                "orientation": {
                    "w": package.orientation.w,
                    "l": package.orientation.l,
                    "h": package.orientation.h,
                },
                "value": package.value,
                "weight": package.weight,
                "isFragile": package.isFragile,
                "eta": package.eta,
            }
            for package in problem.packages.values()
        ],
    }

def problem_to_json(problem: Problem, indent: int = 2) -> str:
    return json.dumps(problem_to_dict(problem), indent=indent)


def problem_from_dict(data: dict[str, Any]) -> Problem:
    if not isinstance(data, dict):
        raise TypeError("problem JSON harus berupa object")

    truck_data = _required_object(data, "truck")
    packages_data = data.get("packages")
    if not isinstance(packages_data, list):
        raise TypeError("packages harus berupa array")

    truck = Truck(
        dimensions=_parse_dimensions(truck_data, "dimensions"),
        maxCapacity=_required(truck_data, "maxCapacity"),
    )

    packages: dict[str, Package] = {}
    for package_data in packages_data:
        if not isinstance(package_data, dict):
            raise TypeError("setiap package harus berupa object")

        package = Package(
            id=_required(package_data, "id"),
            dimensions=_parse_dimensions(package_data, "dimensions"),
            orientation=_parse_orientation(package_data, "orientation"),
            value=_required(package_data, "value"),
            weight=_required(package_data, "weight"),
            isFragile=_required(package_data, "isFragile"),
            eta=_required(package_data, "eta"),
        )
        if package.id in packages:
            raise ValueError(f"duplicate package id: {package.id}")
        packages[package.id] = package

    return Problem(packages=packages, truck=truck)


def _parse_dimensions(data: dict[str, Any], key: str) -> Dimensions:
    dimensions = _required_object(data, key)
    return Dimensions(
        w=_required(dimensions, "w"),
        l=_required(dimensions, "l"),
        h=_required(dimensions, "h"),
    )


def _parse_orientation(data: dict[str, Any], key: str) -> Orientation:
    orientation = _required_object(data, key)
    return Orientation(
        w=_required(orientation, "w"),
        l=_required(orientation, "l"),
        h=_required(orientation, "h"),
    )


def _dimensions_to_dict(dimensions: Dimensions) -> dict[str, int]:
    return {"w": dimensions.w, "l": dimensions.l, "h": dimensions.h}


def _required_object(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = _required(data, key)
    if not isinstance(value, dict):
        raise TypeError(f"{key} harus berupa object")
    return value


def _required(data: dict[str, Any], key: str) -> Any:
    if key not in data:
        raise ValueError(f"field wajib tidak ditemukan: {key}")
    return data[key]