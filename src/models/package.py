from dataclasses import dataclass
from typing import Literal

from src.models.dims import Dimensions

@dataclass
class Package:
    id: str
    dimensions: Dimensions
    orientation: Literal["Horizontal", "Vertical"]
    value: int
    isFragile: bool
    eta: int

    def __post_init__(self):
        valid_options = ["Horizontal", "Vertical"]
        if self.orientation not in valid_options:
            raise ValueError(f"orientation harus salah satu dari {valid_options}")
