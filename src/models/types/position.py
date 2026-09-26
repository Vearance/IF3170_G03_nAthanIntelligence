from dataclasses import dataclass


@dataclass
class Position:
    x: int
    y: int
    z: int

    def __post_init__(self) -> None:
        coordinates = (self.x, self.y, self.z)
        if any(
            isinstance(coordinate, bool) or not isinstance(coordinate, int)
            for coordinate in coordinates
        ):
            raise TypeError("position harus berupa integer")

        if any(coordinate < 0 for coordinate in coordinates):
            raise ValueError("position tidak boleh negatif")
