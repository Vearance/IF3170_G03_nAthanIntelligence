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
        if not isinstance(self.orientation, Orientation):
            raise TypeError("orientation harus berupa instance Orientation")
