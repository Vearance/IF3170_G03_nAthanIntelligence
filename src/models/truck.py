from dataclasses import dataclass

from src.models.types.dims import Dimensions


@dataclass
class Truck:
    dimensions: Dimensions
    maxCapacity: int

    def __post_init__(self) -> None:
        if isinstance(self.maxCapacity, bool) or not isinstance(
            self.maxCapacity, int
        ):
            raise TypeError("maxCapacity harus berupa integer")
        if self.maxCapacity < 0:
            raise ValueError("maxCapacity tidak boleh negatif")
