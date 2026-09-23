from dataclasses import dataclass

from src.models.dims import Dimensions

@dataclass
class Truck:
    dimensions: Dimensions
    maxCapacity: int
    
