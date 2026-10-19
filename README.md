# Tubes 1 AI

Repo ini menggunakan [uv](https://docs.astral.sh/uv/) untuk mengelola environment dan dependency Python.

## Perintah dasar

```bash
# Install/sinkronisasi dependency
uv sync

# Try visualizer
uv run vis
# or
uv run vis test-4

# Run local search and write output/{input}_{algorithm}_state.json
# plus output/{input}_{algorithm}_details.txt
uv run solve
uv run solve test-4 --algorithm hill-climbing
uv run solve test-4 --algorithm simulated-annealing --iterations 1000 --seed 42
uv run solve test-4 --algorithm genetic-algorithm --population-size 50 --iterations 100 --seed 42

# Toggle between initial and final state from a saved snapshot
uv run vis-state test-4-hc.json

# Use a different output directory
uv run solve test-4 --output-dir output/experiments
```

`json_file` accepts an existing path or a filename from `test/input` (with or
without the `.json` suffix), matching `uv run vis`.

```bash
# Tambahkan dependency
uv add <nama-package>
```
