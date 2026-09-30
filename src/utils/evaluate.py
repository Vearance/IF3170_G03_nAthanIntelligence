from dataclasses import replace
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

class Evaluate:

	@staticmethod
	def value(state: State) -> int:
		return sum(
			placed[0].value
			for placed in _placed_packages(state)
		)

	@staticmethod
	def inside_truck(state: State) -> bool:
		truck = state.problem.truck.dimensions
		for package, cells, min_x, min_y, min_z, width, length, height in _placed_packages(state):
			if (
				min_x < 0
				or min_y < 0
				or min_z < 0
				or min_x + width > truck.w
				or min_y + length > truck.l
				or min_z + height > truck.h
			):
				return False

			expected = {
				(x, y, z)
				for z in range(min_z, min_z + height)
				for y in range(min_y, min_y + length)
				for x in range(min_x, min_x + width)
			}
			if cells != expected:
				return False
		return True

	@staticmethod
	def no_floating(state: State) -> bool:
		cells = _cell_map(state)
		for package, _, min_x, min_y, min_z, width, length, _ in _placed_packages(state):
			base = (
				(x, y, min_z)
				for x in range(min_x, min_x + width)
				for y in range(min_y, min_y + length)
			)
			if min_z == 0:
				continue
			if not any(
				(x, y, min_z - 1) in cells
				and cells[(x, y, min_z - 1)] != package.id
				for x, y, _ in base
			):
				return False
		return True

	@staticmethod
	def fragile_not_supporting(state: State) -> bool:
		cells = _cell_map(state)
		for package, _, min_x, min_y, min_z, width, length, height in _placed_packages(state):
			if not package.isFragile:
				continue
			top_z = min_z + height
			for x in range(min_x, min_x + width):
				for y in range(min_y, min_y + length):
					above = cells.get((x, y, top_z))
					if above is not None and above != package.id:
						return False
		return True

	@staticmethod
	def within_capacity(state: State) -> bool:
		return sum(placed[0].weight for placed in _placed_packages(state)) <= (
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
def _placed_packages(
	state: State,
) -> Iterator[tuple[Package, frozenset[tuple[int, int, int]], int, int, int, int, int, int]]:
	cells_by_package: dict[str, set[tuple[int, int, int]]] = {}
	for z, layer in enumerate(state.grid):
		for y, row in enumerate(layer):
			for x, package_id in enumerate(row):
				if package_id is not None:
					cells_by_package.setdefault(package_id, set()).add((x, y, z))

	for package_id, cells in cells_by_package.items():
		package = state.problem.get_package(package_id)
		oriented_package = replace(
			package,
			orientation=state.pkg_orientation.get(package_id, package.orientation),
		)
		min_x = min(x for x, _, _ in cells)
		min_y = min(y for _, y, _ in cells)
		min_z = min(z for _, _, z in cells)
		yield (
			oriented_package,
			frozenset(cells),
			min_x,
			min_y,
			min_z,
			_axis_dimension(oriented_package, "x"),
			_axis_dimension(oriented_package, "y"),
			_axis_dimension(oriented_package, "z"),
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