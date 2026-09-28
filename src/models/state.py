from dataclasses import dataclass

from src.models.problem import Problem
from src.models.types.orientation import Orientation

# define cell and grid types
Cell = str | None  # None means empty
Grid = list[list[list[Cell]]]  # [z][y][x]


@dataclass
class State:
    problem: Problem  # does not change
    grid: Grid
    pkg_orientation: dict[str, Orientation]  # local change for package orientation
    out_pkg: set[str]  # local change for outside packages
    value: int  # acts like a cache; need to be recalculated
