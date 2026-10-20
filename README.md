# Tubes 1 AI

Repo ini menggunakan [uv](https://docs.astral.sh/uv/) untuk mengelola environment dan dependency Python.

## Perintah dasar

```bash
# uv basic command
# Install/sinkronisasi dependency
uv sync
# Tambahkan dependency
uv add <nama-package>

# Try visualizer; untuk dynamic GUI
# Kalau mau lihat initial/final state aja, gunakan 'solve' dan 'vis-state' instead
uv run vis

# Run local search and write output state in json + txt
uv run solve
uv run solve test-4 --algorithm hill-climbing
uv run solve test-4 --algorithm hill-climbing-sideways --max-sideways-moves 20
uv run solve test-4 --algorithm hill-climbing-random-restart --max-restarts 10 --seed 42
uv run solve test-4 --algorithm simulated-annealing --iterations 1000 --seed 42
uv run solve test-4 --algorithm genetic-algorithm --population-size 50 --iterations 100 --seed 42

# Toggle between initial and final state from a saved snapshot
uv run vis-state test-4-hc.json

# Use a different output directory
uv run solve test-4 --output-dir output/experiments
```

## Struktur

```text
.
├── src/
│   ├── algo/
│   │   ├── genetic.py
│   │   ├── hill_climb.py
│   │   ├── hill_climb_sideway.py
│   │   ├── hill_climb_random_restart.py
│   │   └── simulated_annealing.py
│   ├── models/
│   │   ├── package.py
│   │   ├── problem.py
│   │   ├── result.py
│   │   ├── state.py
│   │   ├── truck.py
│   │   └── types/
│   ├── utils/
│   │   ├── evaluate.py
│   │   ├── move.py
│   │   ├── parser.py
│   │   ├── plotting.py
│   │   ├── random_state.py
│   │   └── snapshot.py
│   ├── main.py
│   └── visualizer.py
├── test/
│   ├── input/
│   └── scripts/
├── docs/
│   └── laporan.md
├── output/
├── pyproject.toml
├── uv.lock
└── README.md
```

Komponen utama proyek dapat dikelompokkan sebagai berikut:

-*Model: merepresentasikan truk, paket, problem, orientasi, posisi, dan state.
- Algoritma: mencari konfigurasi pengisian truk dengan beberapa metode local search.
- Utility: membaca input, membuat state awal, melakukan perpindahan paket, memvalidasi state, dan membuat output.
- Antarmuka: `main.py` untuk menjalankan solver dan `visualizer.py` untuk menampilkan proses secara interaktif.


## Deskripsi Sistem Singkat

Sistem ini menyelesaikan masalah optimasi pengisian paket ke dalam truk peti kemas. Masukan sistem berupa file JSON yang berisi dimensi truk, kapasitas maksimum, serta daftar paket. Setiap paket memiliki dimensi, orientasi, nilai, berat, status fragile, dan ETA.

Tujuan sistem adalah memaksimalkan nilai total paket yang berhasil dimasukkan ke truk. Paket yang tidak dapat dimasukkan tetap dicatat sebagai paket di luar truk. Sebuah konfigurasi dianggap valid apabila memenuhi batas dimensi truk, tidak memiliki tabrakan antarpaket, tidak membuat paket melayang, tidak menempatkan paket di atas paket fragile, dan tidak melebihi kapasitas truk.

Representasi isi truk menggunakan grid tiga dimensi dengan format `grid[z][y][x]`. Selain grid, setiap `State` menyimpan orientasi aktual paket, kumpulan paket di luar truk, dan nilai objektif yang dihitung dari jumlah nilai paket yang berada di dalam truk.

Algoritma yang tersedia adalah:

1. Steepest-Ascent Hill Climbing
2. Hill Climbing with Sideways Move
3. Random Restart Hill Climbing
4. Simulated Annealing
5. Genetic Algorithm

Sistem juga menyediakan visualizer 3D berbasis Viser untuk menampilkan truk, paket, orientasi, nilai objektif, dan perubahan state selama pencarian.

## Anggota

| No. | Nama | NIM |
|---:|---|---|
| 1 | Bernhard Aprillio Pramana | 13524074 |
| 2 | Jennifer Khang | 13524110 |
| 3 | Nathaniel Christian | 13524122 |