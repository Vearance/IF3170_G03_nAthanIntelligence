# Tubes 1 AI

Repo ini menggunakan [uv](https://docs.astral.sh/uv/) untuk mengelola environment dan dependency Python.

## Perintah dasar

```bash
# Install/sinkronisasi dependency
uv sync

# Jalankan visualisasi package-truck
uv run python src/main.py test/input/test-1.json

# Buka URL yang tampil di terminal (default: http://localhost:8080)

# Tambahkan dependency
uv add <nama-package>
```
