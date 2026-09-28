from dataclasses import dataclass

from src.models.package import Package
from src.models.truck import Truck


@dataclass
class Problem:
    packages: dict[str, Package]  # packages catalog
    truck: Truck

    def __post_init__(self) -> None:
        for package_id, package in self.packages.items():
            if package_id != package.id:
                raise ValueError(
                    "key package catalog harus sama dengan id package"
                )

    def get_package(self, package_id: str) -> Package:
        try:
            return self.packages[package_id]
        except KeyError as err:
            raise KeyError(f"package tidak ditemukan: {package_id}") from err
