from dataclasses import dataclass, replace

from src.models.problem import Problem
from src.models.types.orientation import Orientation
from src.models.types.position import Position

# define cell and grid types
Cell = str | None  # None means empty; ini isinya package id
Grid = list[list[list[Cell]]]  # [z][y][x]

# Note: sebuah grid merupakan representasi sebuah isi truck pada suatu state


@dataclass
class State:
    problem: Problem  # does not change
    grid: Grid
    pkg_orientation: dict[str, Orientation]  # local change for package orientation
    out_pkg: set[str]  # local change for outside packages
    value: int  # acts like a cache; need to be recalculated


    # create empty state
    # how to use -> var = State.empty(problem)
    @classmethod
    def empty(cls, problem: Problem) -> "State":
        dimensions = problem.truck.dimensions
        grid: Grid = [
            [[None for _ in range(dimensions.w)] for _ in range(dimensions.l)]
            for _ in range(dimensions.h)
        ]
        pkg_orientation = {
            package_id: replace(package.orientation)
            for package_id, package in problem.packages.items()
        }

        return cls(
            problem=problem,
            grid=grid,
            pkg_orientation=pkg_orientation,
            out_pkg=set(problem.packages),
            value=0,
        )

    def __post_init__(self) -> None:
        self.validate_grid()
        package_ids = set(self.problem.packages)
    
        if set(self.pkg_orientation) != package_ids:
            raise ValueError(
                "pkg_orientation harus memiliki semua package id tepat satu kali"
            )
        if not self.out_pkg <= package_ids:
            raise ValueError("out_pkg berisi package id yang tidak dikenal")
    
        for package_id, orientation in self.pkg_orientation.items():
            if not isinstance(orientation, Orientation):
                raise TypeError(
                    f"orientation package {package_id} harus berupa Orientation"
                )
    
        package_ids_in_grid = self.package_ids_in_grid()
        if package_ids_in_grid & self.out_pkg:
            raise ValueError(
                "package tidak boleh berada di grid dan out_pkg sekaligus"
            )
        if package_ids_in_grid | self.out_pkg != package_ids:
            raise ValueError(
                "setiap package harus berada di grid atau out_pkg"
            )
    
        self.value = self.recalculate_value()

    def copy(self) -> "State":
        grid = [[row[:] for row in layer] for layer in self.grid]
        pkg_orientation = {
            package_id: replace(orientation)
            for package_id, orientation in self.pkg_orientation.items()
        }
        return State(
            problem=self.problem,
            grid=grid,
            pkg_orientation=pkg_orientation,
            out_pkg=set(self.out_pkg),
            value=self.value,
        )

    # return all id package (set) yang ada di grid
    def package_ids_in_grid(self) -> set[str]:
        package_ids: set[str] = set()
        known_package_ids = set(self.problem.packages)
        for layer in self.grid:
            for row in layer:
                for package_id in row:
                    if package_id is None:
                        continue
                    if not isinstance(package_id, str):
                        raise TypeError("grid hanya boleh berisi package id atau None")
                    if package_id not in known_package_ids:
                        raise ValueError(
                            f"package id tidak dikenal di grid: {package_id}"
                        )
                    package_ids.add(package_id)
        return package_ids

    # calculate value untuk semua package
    def recalculate_value(self) -> int:
        return sum(
            self.problem.packages[package_id].value
            for package_id in self.package_ids_in_grid()
        )

    # return 'Cell' pada position tertentu
    def cell_at(self, position: Position) -> Cell:
        dimensions = self.problem.truck.dimensions
        if not (
            0 <= position.x < dimensions.w
            and 0 <= position.y < dimensions.l
            and 0 <= position.z < dimensions.h
        ):
            raise IndexError(f"position berada di luar grid: {position}")
        return self.grid[position.z][position.y][position.x]

    def validate_grid(self) -> None:
        dimensions = self.problem.truck.dimensions
        if len(self.grid) != dimensions.h:
            raise ValueError("ukuran sumbu z grid tidak sesuai dengan truck")
    
        for layer in self.grid:
            if len(layer) != dimensions.l:
                raise ValueError("ukuran sumbu y grid tidak sesuai dengan truck")
            for row in layer:
                if len(row) != dimensions.w:
                    raise ValueError("ukuran sumbu x grid tidak sesuai dengan truck")
