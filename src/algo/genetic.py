from __future__ import annotations

# !!! Hapus ini ketika ingin melakukan PR:
# file ini tidak mengecek validasi input, asumsinya input selalu sesuai kontrak;
# selain itu, beberapa funsgi seharusnya bisa di-utils, namun karena hanya cocok untuk GA,
# fungsi tidak dipindahkan
from collections.abc import Callable
from dataclasses import replace
from random import Random
from time import perf_counter

from src.models.result import Result
from src.models.state import State
from src.models.types.position import Position
from src.utils.evaluate import Evaluate
from src.utils.move import (
    _package_anchor,
    try_move,
    try_place,
    try_rotate,
    try_swap,
)
from src.utils.random_state import random_initial_state, random_position

MUTATION_RATE = 0.2


# func: menjalankan genetic_algorithm dengan State sebagai individu
# pengecekan error hanya dilakukan pada parameter yang akan diubah
# untuk tiap param yang berubah, sudah dijadikan lainnya dengan nilai default sendiri
def genetic_algorithm(
    initial_state: State,
    *,
    population_size: int = 50,
    max_iterations: int = 100,
    seed: int | None = None,
    on_iteration: Callable[[State, int], None] | None = None,
) -> Result:
    
    # cek populasi
    if isinstance(population_size, bool) or not isinstance(population_size, int):
        raise TypeError("population_size harus berupa integer")
    if population_size < 2:
        raise ValueError("population_size minimal 2")

    # cek parameter iterasi
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise TypeError("max_iterations harus berupa integer")
    if max_iterations < 0:
        raise ValueError("max_iterations tidak boleh negatif")

    started_at = perf_counter()
    # copy juga memeriksa invariant struktur dan pembagian paket dalam/luar
    initial = initial_state.copy()
    if not Evaluate.is_valid(initial):
        raise ValueError("initial_state harus valid")

    rng = Random(seed)
    population = _initialize_population(initial, population_size, rng)
    initial_best = max(population, key=lambda state: state.value).copy()

    # simpen statistik
    max_value_history: list[int] = []
    mean_value_history: list[float] = []
    _record_statistics(population, max_value_history, mean_value_history)

    # tiap generasi dibangkitkan
    for generation in range(1, max_iterations + 1):
        next_population: list[State] = []

        while len(next_population) < population_size:
            first_parent = _select_parent(population, rng)
            second_parent = _select_parent(population, rng)
            child = _crossover(first_parent, second_parent, rng)

            if rng.random() < MUTATION_RATE:
                child = _mutate(child, rng)
            next_population.append(child)

        population = next_population
        _record_statistics(population, max_value_history, mean_value_history)
        if on_iteration is not None:
            best = max(population, key=lambda state: state.value)
            on_iteration(best.copy(), generation)

    best = max(population, key=lambda state: state.value)
    return Result(
        initial_state=initial_best,
        state=best.copy(),
        value_history=max_value_history,
        iterations=max_iterations,
        duration_seconds=perf_counter() - started_at,
        metrics={
            "population_size": population_size,
            "max_iterations": max_iterations,
            "seed": seed,
            "mean_value_history": mean_value_history,
        },
    )

# func: generator untuk populasi awal genetic_algorithm
def _initialize_population(
    initial_state: State,
    population_size: int,
    rng: Random,
) -> list[State]:
    return [
        random_initial_state(initial_state.problem, rng)
        for _ in range(population_size)
    ]

# func: memilih satu parent dari populasi
def _select_parent(
    population: list[State],
    rng: Random,
) -> State:
    if not population:
        raise ValueError("population tidak boleh kosong")

    # fitness dicatat ulang untuk seluruh populasi sebelum seleksi generasi ini
    weights = [state.value for state in population]
    if sum(weights) == 0:
        return rng.choice(population)
    return rng.choices(population, weights=weights, k=1)[0]


# func: membentuk State anak langsung dari dua parent
def _crossover(
    p1: State,
    p2: State,
    rng: Random,
) -> State:
    package_ids = sorted(p1.problem.packages)
    # tidak ada titik potong internal untuk kromosom kosong atau satu gen.
    if len(package_ids) < 2:
        return p1.copy()
    cut = rng.randrange(1, len(package_ids))

    # init
    child = State.empty(p1.problem)
    proposals: list[tuple[str, Position]] = []

    # satu gen = posisi/status dalam-luar dan orientasi satu paket.
    # prefix diwarisi dari p1, suffix dari p2 pada satu titik potong.
    for index, package_id in enumerate(package_ids):
        parent = p1 if index < cut else p2
        child.pkg_orientation[package_id] = replace(
            parent.pkg_orientation[package_id]
        )

        # fiksasi parent
        position = _package_anchor(parent, package_id)
        
        if position is not None:
            proposals.append((package_id, position))

    # shuffle sebelum stable sort agar prioritas pada z yang sama tidak bias ID
    rng.shuffle(proposals)
    proposals.sort(key=lambda proposal: proposal[1].z)

    # penempatan ulang sekaligus cek invarian paket
    for package_id, position in proposals:
        candidate = try_place(child, package_id, position)
        if candidate is not None:
            child = candidate

    return child


# func: mencoba satu mutasi pada individu
def _mutate(state: State, rng: Random) -> State:
    # init
    original = state.copy()
    package_ids = sorted(state.problem.packages)
    if not package_ids:
        return original

    # menentukan aksi mutasi
    operations = ["move", "rotate"]
    if len(package_ids) >= 2:
        operations.append("swap")
    operation = rng.choice(operations)
    package_id = rng.choice(package_ids)

    # setelah rng, langsung cek action
    if operation == "swap":
        first_id, second_id = rng.sample(package_ids, 2)
        candidate = try_swap(original, first_id, second_id)
    elif operation == "rotate":
        candidate = try_rotate(original, package_id, rng.choice(("x", "y", "z")))

    # kalau bukan rotate dan swap, lakukan move
    else: 
        if package_id not in original.out_pkg and rng.random() < 0.2:
            candidate = try_move(original, package_id, None)
        else:
            position = random_position(original, package_id, rng)
            if position is None:
                return original
            candidate = try_move(original, package_id, position)

    if candidate is None:
        return original

    return candidate

# !!! Utils untuk GA, tidak di masukkan ke folder utils karena ini hanya dipakai di sini :3

# func: mencatat statistik populasi pada satu generasi
def _record_statistics(
    population: list[State],
    max_value_history: list[int],
    mean_value_history: list[float],
) -> None:
    if not population:
        raise ValueError("population tidak boleh kosong")
    for state in population:
        state.value = Evaluate.value(state) # fitness function
    values = [state.value for state in population]
    max_value_history.append(max(values))
    mean_value_history.append(sum(values) / len(values))
