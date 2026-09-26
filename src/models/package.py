from dataclasses import dataclass

from src.models.types.dims import Dimensions
from src.models.types.orientation import Orientation


@dataclass
class Package:
    id: str
    dimensions: Dimensions
    orientation: Orientation
    value: int
    weight: float
    isFragile: bool
    # eta: int  # should be time format; unused attribute

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("package id harus berupa string non-empty")
        if not isinstance(self.orientation, Orientation):
            raise TypeError("orientation harus berupa instance Orientation")
        if isinstance(self.value, bool) or not isinstance(self.value, int):
            raise TypeError("package value harus berupa integer")
        if self.value < 0:
            raise ValueError("package value tidak boleh negatif")
        if isinstance(self.weight, bool) or not isinstance(
            self.weight, (int, float)
        ):
            raise TypeError("package weight harus berupa angka")
        if self.weight < 0:
            raise ValueError("package weight tidak boleh negatif")
        if not isinstance(self.isFragile, bool):
            raise TypeError("isFragile harus berupa boolean")
