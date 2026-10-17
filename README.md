# Tubes 1 AI

Repo ini menggunakan [uv](https://docs.astral.sh/uv/) untuk mengelola environment dan dependency Python.

## Perintah dasar

```bash
# Install/sinkronisasi dependency
uv sync

# Try visualizer
uv run python src/visualizer.py test/input/test-1.json

# Try test
uv run python -m unittest discover -s test/scripts -p "test_*.py"

# Tambahkan dependency
uv add <nama-package>
```
