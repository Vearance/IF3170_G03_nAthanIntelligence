from matplotlib.figure import Figure

from src.models.result import Result

# func: membuat grafik maksimum dan rata-rata populasi GA
def create_genetic_plot(result: Result, title: str = "Genetic Algorithm") -> Figure:
    mean_history = result.metrics["mean_value_history"]
    if not isinstance(mean_history, list):
        raise ValueError("mean_value_history harus berupa list")
    if len(mean_history) != len(result.value_history):
        raise ValueError("history maksimum dan rata-rata harus sama panjang")

    figure = Figure(figsize=(8, 4.5), dpi=160)
    axis = figure.subplots()
    generations = range(len(result.value_history))
    single_generation = len(result.value_history) == 1
    axis.plot(generations, result.value_history, color="#386641", label="Maksimum",
              marker="o" if single_generation else None)
    axis.plot(generations, mean_history, color="#386641", linestyle="--", label="Rata-rata",
              marker="s" if single_generation else None)
    axis.set(title=title, xlabel="Generasi", ylabel="Nilai objective function")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    return figure
