from dataclasses import dataclass, replace
from typing import Iterator

from src.models.package import Package
from src.models.state import State

"""
	Evaluator akan memeriksa sebuah state:
		1. Tidak menembus dinding truk
			x >= 0
			x + width <= truck.width

			(Berlaku juga untuk y dan z).
			
		2. Tidak bertabrakan dengan paket lain (tidak boleh ada dua box di satu titik koordinat posisi)
		
		3. Tidak melayang
		
		4. Paket di lantai truk valid.
			Kalau tidak di lantai, harus memiliki support dari paket lain.
			Support terjadi jika bagian atas paket lain tepat berada di bawah alas paket.
			
		5. Paket yang isFragile==True tidak boleh menjadi support
			
		6. Tidak melebihi kapasitas
			total weight paket di truk <= maxCapacity

	NB: Paket di luar truk tidak ikut pengecekan collision dan tidak dihitung dalam kapasitas.
"""

@dataclass(frozen=True)
class _PlacedPackage:
	package: Package
	cells: frozenset[tuple[int, int, int]]
	min_x: int
	min_y: int
	min_z: int
	width: int
	length: int
	height: int


@dataclass
class Evaluate:

	@staticmethod
	def value(state: State) -> int:
		return sum(
			placed.package.value
			for placed in _placed_packages(state)
		)

	@staticmethod
	def inside_truck(state: State) -> bool:
		truck = state.problem.truck.dimensions
		for placed in _placed_packages(state):
			if (
				placed.min_x < 0
				or placed.min_y < 0
				or placed.min_z < 0
				or placed.min_x + placed.width > truck.w
				or placed.min_y + placed.length > truck.l
				or placed.min_z + placed.height > truck.h
			):
				return False

			expected = {
				(x, y, z)
				for z in range(placed.min_z, placed.min_z + placed.height)
				for y in range(placed.min_y, placed.min_y + placed.length)
				for x in range(placed.min_x, placed.min_x + placed.width)
			}
			if placed.cells != expected:
				return False
		return True

	@staticmethod
	def no_floating(state: State) -> bool:
		cells = _cell_map(state)
		for placed in _placed_packages(state):
			base = (
				(x, y, placed.min_z)
				for x in range(placed.min_x, placed.min_x + placed.width)
				for y in range(placed.min_y, placed.min_y + placed.length)
			)
			if placed.min_z == 0:
				continue
			if not any(
				(x, y, placed.min_z - 1) in cells
				and cells[(x, y, placed.min_z - 1)] != placed.package.id
				for x, y, _ in base
			):
				return False
		return True

	@staticmethod
	def fragile_not_supporting(state: State) -> bool:
		cells = _cell_map(state)
		packages = state.problem.packages
		for placed in _placed_packages(state):
			if not placed.package.isFragile:
				continue
			top_z = placed.min_z + placed.height
			for x in range(placed.min_x, placed.min_x + placed.width):
				for y in range(placed.min_y, placed.min_y + placed.length):
					above = cells.get((x, y, top_z))
					if above is not None and above != placed.package.id:
						return False
		return True

	@staticmethod
	def within_capacity(state: State) -> bool:
		return sum(placed.package.weight for placed in _placed_packages(state)) <= (
			state.problem.truck.maxCapacity
		)

	@staticmethod
	def is_valid(state: State) -> bool:
		return (
			Evaluate.inside_truck(state)
			and Evaluate.no_floating(state)
			and Evaluate.fragile_not_supporting(state)
			and Evaluate.within_capacity(state)
		)


def _placed_packages(state: State) -> Iterator[_PlacedPackage]:
	cells_by_package: dict[str, set[tuple[int, int, int]]] = {}
	for z, layer in enumerate(state.grid):
		for y, row in enumerate(layer):
			for x, package_id in enumerate(row):
				if package_id is not None:
					cells_by_package.setdefault(package_id, set()).add((x, y, z))

	for package_id, cells in cells_by_package.items():
		package = state.problem.get_package(package_id)
		orientation = state.pkg_orientation.get(package_id, package.orientation)
		oriented_package = replace(package, orientation=orientation)
		min_x = min(x for x, _, _ in cells)
		min_y = min(y for _, y, _ in cells)
		min_z = min(z for _, _, z in cells)
		yield _PlacedPackage(
			package=oriented_package,
			cells=frozenset(cells),
			min_x=min_x,
			min_y=min_y,
			min_z=min_z,
			width=_axis_dimension(oriented_package, "x"),
			length=_axis_dimension(oriented_package, "y"),
			height=_axis_dimension(oriented_package, "z"),
		)


def _axis_dimension(package: Package, axis: str) -> int:
	for dimension in ("w", "l", "h"):
		if getattr(package.orientation, dimension) == axis:
			return getattr(package.dimensions, dimension)
	raise ValueError(f"orientation tidak memiliki sumbu {axis}")


def _cell_map(state: State) -> dict[tuple[int, int, int], str]:
	return {
		(x, y, z): package_id
		for z, layer in enumerate(state.grid)
		for y, row in enumerate(layer)
		for x, package_id in enumerate(row)
		if package_id is not None
	}