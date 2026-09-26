from dataclasses import dataclass


@dataclass
class Dimensions:
    w: int
    l: int
    h: int

    def __post_init__(self) -> None:
        dimensions = (self.w, self.l, self.h)
        if any(
            isinstance(dimension, bool) or not isinstance(dimension, int)
            for dimension in dimensions
        ):
            raise TypeError("dimensions harus berupa integer")

        if any(dimension <= 0 for dimension in dimensions):
            raise ValueError("dimensions harus lebih besar dari nol")
