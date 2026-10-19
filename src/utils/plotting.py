from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from src.models.result import Result


# func(general): memilih plot sesuai algoritma 
def create_search_plots(result: Result, algorithm: str) -> list[Figure]:
    if algorithm in {"genetic-algorithm", "Genetic Algorithm"}:
        return [create_genetic_plot(result)]
    if algorithm in {"simulated-annealing", "Simulated Annealing"}:
        figures = [create_objective_plot(result, "Simulated Annealing - Current Value")]
        if "acceptance_probability_history" in result.metrics:
            figures.append(create_annealing_plot(result))
        return figures
    return [create_objective_plot(result, "Steepest-Ascent Hill-Climbing")]


# func(general): plot objective function dari state awal sampai iterasi terakhir
def create_objective_plot(
    result: Result, title: str
) -> Figure:
    fig = Figure(figsize=(8, 4.5), dpi=160)
    ax = fig.subplots()
    ax.plot(result.value_history, marker=".")
    ax.set(title=title, xlabel="Iterasi", ylabel="Nilai objective function")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


# func(GA): plot nilai objective function maksimum dan rata-rata dari populasi terhadap banyak iterasi 
def create_genetic_plot(result: Result, title: str = "Genetic Algorithm") -> Figure:
    mean_history = result.metrics.get("mean_value_history")
    
    if not isinstance(mean_history, list) or len(mean_history) != len(result.value_history):
        raise ValueError("mean_value_history harus sama panjangnya dengan value_history")

    fig = create_objective_plot(result, title)
    ax = fig.axes[0]
    ax.lines[0].set_label("Maksimum")
    ax.plot(mean_history, "--", label="Rata-rata", marker=".")
    ax.legend()
    return fig


# func(SA): plot probabilitas penerimaan kandidat pada tiap iterasi
def create_annealing_plot(
    result: Result, title: str = "Simulated Annealing - Probabilitas Penerimaan"
) -> Figure:
    history = result.metrics.get("acceptance_probability_history")
    if not isinstance(history, list) or len(history) != result.iterations:
        raise ValueError("acceptance_probability_history harus sepanjang iterations")

    fig = Figure(figsize=(8, 4.5), dpi=160)
    ax = fig.subplots()
    # NaN memutus garis pada iterasi yang tidak memiliki kandidat valid
    probabilities = [float("nan") if value is None else value for value in history]
    ax.plot(range(1, result.iterations + 1), probabilities, marker=".")
    ax.set(
        title=title,
        xlabel="Iterasi",
        ylabel="Probabilitas penerimaan",
        ylim=(-0.05, 1.05),
    )
    if not history or all(value is None for value in history):
        ax.text(0.5, 0.5, "Belum ada kandidat valid", ha="center", transform=ax.transAxes)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig
