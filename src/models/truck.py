from dataclasses import dataclass

from src.models.types.dims import Dimensions


@dataclass
class Truck:
    dimensions: Dimensions
    maxCapacity: int
