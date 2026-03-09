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

### Prerequisites
- A Google account with access to [Google Colab](https://colab.research.google.com/)
- No local installation required — all dependencies (Python 3.12, NumPy, Numba, CuPy, Matplotlib) are pre-installed on Colab

### Execution Steps

1. **Upload the notebook**
   Go to [Google Colab](https://colab.research.google.com/) → File → Upload notebook → select `p1/imc06_floyd_warshall_experiments.ipynb`

2. **Select the GPU runtime**
   Runtime → Change runtime type → set Hardware accelerator to **T4 GPU** → Save

3. **Run all cells in order**
   Runtime → Run all (or `Ctrl+F9`)

### What Each Cell Does

| Cell | Description | Expected Output |
|------|-------------|-----------------|
| **Cell 1** | Imports, constants, graph generator | Prints a sample 4-node graph; confirms CUDA is available |
| **Cell 2** | Variant 1 — NumPy CPU baseline | Prints shortest-path result for a 4-node test graph |
| **Cell 3** | Variant 2 — Naive Numba CUDA kernel (16x16 blocks) | Prints GPU result + max diff vs CPU (should be 0.0) |
| **Cell 4** | Variant 3 — Tiled Numba CUDA kernel (32x32 blocks + shared memory) | Prints GPU result + max diff vs CPU (should be 0.0) |
| **Cell 5** | Variant 4 — CuPy GPU broadcasting | Prints GPU result + max diff vs CPU (should be 0.0) |
| **Cell 6** | Benchmarks — runs all 4 variants at n = 256, 512, 1024, 2048 | Prints runtimes, correctness diffs, and speedups for each size |
| **Cell 7** | Results — summary table, operations count, and performance plots | Prints formatted results table; saves plot as `floyd_warshall_results.png` |

### Expected Runtime
- **Total execution time:** ~2–3 minutes on a Tesla T4 GPU (most time is spent on n=2048)
- The first GPU kernel launch may take a few extra seconds due to Numba JIT compilation

### Verifying Correctness
Every GPU variant is checked against the CPU baseline. The `diff` column in Cell 6 output should show **0.000000** for all sizes and variants, confirming that all implementations produce identical shortest-path results.

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

## Video Recordings

- **Part 0 (5-min Theory/PCAM Walkthrough):** [Video Link](https://myharrisburgu-my.sharepoint.com/:v:/g/personal/kfernandes_my_harrisburgu_edu/IQAug-7lPTtNQ61X7fIhs012AeY6_gsCXGclUJ7-bgN5Vbc)
- **Part 3 (10-min Track A Walkthrough):** [Video Link](https://myharrisburgu-my.sharepoint.com/:v:/g/personal/kfernandes_my_harrisburgu_edu/IQCV7EVDJVGeRohjBfbMxT8hAUTspwo_Blqf_85ye0_dY7Y)

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
