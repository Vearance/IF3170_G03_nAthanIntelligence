from matplotlib.figure import Figure
from src.models.result import Result

def create_genetic_plot(result: Result, title: str = "Genetic Algorithm") -> Figure:
    mean_history = result.metrics.get("mean_value_history")
    
    if not isinstance(mean_history, list) or len(mean_history) != len(result.value_history):
        raise ValueError("mean_value_history harus sama panjangnya dengan value_history")

    fig = Figure(figsize=(8, 4.5), dpi=160)
    ax = fig.subplots()
    
    single_gen = len(result.value_history) == 1
    marker_max, marker_mean = ("o", "s") if single_gen else (None, None)

    ax.plot(result.value_history, label="Maksimum", marker=marker_max)
    ax.plot(mean_history, "--", label="Rata-rata", marker=marker_mean)
    ax.set(title=title, xlabel="Iterasi", ylabel="Nilai objective function")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()

    return fig