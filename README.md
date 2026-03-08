# IMC06 - Floyd's Algorithm (Track A)

**Author:** Kenneth Peter Fernandes  
**University:** Harrisburg University of Science and Technology  
**Course:** CISC 719 - Spring 2026  
**Instructor:** Prof. Majid Shaalan, Ph.D.

## Overview

This project implements and compares multiple versions of the Floyd-Warshall all-pairs shortest path algorithm, starting from two reference implementations (R1 and R2) and extending them with GPU modernizations. All experiments were run on Google Colab with a Tesla T4 GPU.

## Project Structure

```
.
├── README.md
├── p0/                              # Part 0 - Preparation Notes
│   ├── part0_notes.tex              # LaTeX source (PCAM notes, R1/R2 analysis, questions)
│   └── part0_notes.pdf              # Compiled PDF
│
├── p1/                              # Part 1 - Track A Experiments & Report
│   ├── imc06_floyd_warshall_experiments.ipynb   # Main experiment notebook (run on Colab)
│   ├── benchmarks_plottings.png     # Runtime and speedup plots from experiments
│   ├── modernization_report.tex     # LaTeX source for the modernization report
│   └── modernization_report.pdf     # Compiled PDF (2-3 page critique)
│
└── references/                      # Reference materials
    ├── r1_floyd_warshall_cuda.cu    # R1: C++/CUDA reference implementation
    ├── r2_floyd_warshall_fwp_cuda.ipynb  # R2: NumPy + Numba CUDA notebook
    ├── Part0_Understanding_Guide.md # Study notes for understanding PCAM & Floyd-Warshall
    └── chapter 6/                   # Quinn Chapter 6 screenshots
```

## How to Run the Experiments

### Requirements
- Google Colab (free tier with GPU runtime)
- Python 3.10+, NumPy, Numba, CuPy, Matplotlib (all pre-installed on Colab)

### Steps
1. Upload `p1/imc06_floyd_warshall_experiments.ipynb` to Google Colab
2. Set runtime to **GPU** (Runtime > Change runtime type > T4 GPU)
3. Run all cells in order

The notebook runs four variants at problem sizes n = 256, 512, 1024, 2048 and produces:
- Correctness checks (max diff vs CPU should be 0.0)
- Timing table with speedups
- Two plots saved as `floyd_warshall_results.png`

## Implementations

| Variant | Description | Source | Block Size |
|---------|-------------|--------|------------|
| 1 | NumPy CPU baseline (broadcasting) | From R2 | N/A (single core) |
| 2 | Naive Numba CUDA kernel | From R2 | 16x16 |
| 3 | Tiled Numba CUDA kernel + shared memory | **Our extension** | 32x32 |
| 4 | CuPy GPU broadcasting | **Our extension** | Auto (CuPy managed) |

## Key Results (Tesla T4, Google Colab)

| n | CPU (ms) | GPU Naive (ms) | GPU Tiled (ms) | CuPy (ms) | Best Speedup |
|---|----------|---------------|----------------|-----------|-------------|
| 256 | 15.4 | 8.4 | 9.0 | 10.6 | 1.82x (Naive) |
| 512 | 207.8 | 25.1 | 24.2 | 32.7 | 8.57x (Tiled) |
| 1024 | 1,341.7 | 108.8 | 69.5 | 92.6 | 19.30x (Tiled) |
| 2048 | 13,607.6 | 526.9 | 367.9 | 629.9 | 36.99x (Tiled) |

## How to Compile the LaTeX Reports

```bash
cd p0 && pdflatex part0_notes.tex
cd p1 && pdflatex modernization_report.tex
```

Note: `modernization_report.tex` includes `benchmarks_plottings.png`, so the PNG must be in the same directory when compiling.

## File-to-Experiment Mapping

- **Variant 1 (CPU baseline):** `imc06_floyd_warshall_experiments.ipynb`, Cell 2 — `floyd_warshall_cpu()`
- **Variant 2 (GPU naive):** `imc06_floyd_warshall_experiments.ipynb`, Cell 3 — `naive_floyd_kernel()` + `floyd_warshall_gpu_naive()`
- **Variant 3 (GPU tiled):** `imc06_floyd_warshall_experiments.ipynb`, Cell 4 — `tiled_floyd_kernel()` + `floyd_warshall_gpu_tiled()`
- **Variant 4 (CuPy):** `imc06_floyd_warshall_experiments.ipynb`, Cell 5 — `floyd_warshall_cupy()`
- **Benchmarks:** Cell 6
- **Results & Plots:** Cell 7
