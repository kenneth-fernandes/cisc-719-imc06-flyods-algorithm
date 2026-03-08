// floyd_warshall_cuda.cu
//
// Single-file reference implementation for Floyd–Warshall on CPU and CUDA GPU.
// - Builds with: nvcc -O3 -std=c++17 floyd_warshall_cuda.cu -o floyd
// - Runs: ./floyd [n]
//
// This code:
//   * Generates a random dense graph with n vertices
//   * Runs Floyd–Warshall on CPU and GPU
//   * Checks correctness (max absolute diff) and prints timings/speedup

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <limits>
#include <random>
#include <iostream>

constexpr float INF = 1e20f;

inline int idx(int i, int j, int n) {
    return i * n + j;
}

void init_random_graph(std::vector<float>& D, int n,
                       float edge_prob = 0.6f,
                       float max_w = 10.0f,
                       unsigned seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> prob(0.0f, 1.0f);
    std::uniform_real_distribution<float> wdist(1.0f, max_w);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                D[idx(i,j,n)] = 0.0f;
            } else if (prob(gen) < edge_prob) {
                D[idx(i,j,n)] = wdist(gen);
            } else {
                D[idx(i,j,n)] = INF;
            }
        }
    }
}

// ----------------------------- CPU baseline -----------------------------

void floyd_warshall_cpu(std::vector<float>& D, int n) {
    for (int k = 0; k < n; ++k) {
        for (int i = 0; i < n; ++i) {
            float Dik = D[idx(i,k,n)];
            for (int j = 0; j < n; ++j) {
                float alt = Dik + D[idx(k,j,n)];
                float& Dij = D[idx(i,j,n)];
                if (alt < Dij) Dij = alt;
            }
        }
    }
}

// ----------------------------- CUDA kernel ------------------------------

__global__
void floyd_step_kernel(float* D, int n, int k) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= n || j >= n) return;

    int ik = i * n + k;
    int kj = k * n + j;
    int ij = i * n + j;

    float Dik = D[ik];
    float Dkj = D[kj];
    float Dij = D[ij];
    float alt = Dik + Dkj;
    if (alt < Dij) {
        D[ij] = alt;
    }
}

void floyd_warshall_cuda(std::vector<float>& D, int n) {
    float* d_D = nullptr;
    size_t bytes = sizeof(float) * n * n;
    cudaMalloc(&d_D, bytes);
    cudaMemcpy(d_D, D.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x,
              (n + block.y - 1) / block.y);

    for (int k = 0; k < n; ++k) {
        floyd_step_kernel<<<grid, block>>>(d_D, n, k);
    }
    cudaDeviceSynchronize();

    cudaMemcpy(D.data(), d_D, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_D);
}

// ----------------------------- Utilities --------------------------------

double elapsed_ms(const std::chrono::high_resolution_clock::time_point& t0,
                  const std::chrono::high_resolution_clock::time_point& t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

float max_abs_diff(const std::vector<float>& A,
                   const std::vector<float>& B) {
    float m = 0.0f;
    for (size_t i = 0; i < A.size(); ++i) {
        float a = A[i], b = B[i];
        if ((a >= INF/2 && b >= INF/2)) continue; // both "inf"
        float d = fabsf(a - b);
        if (d > m) m = d;
    }
    return m;
}

// ------------------------------- main -----------------------------------

int main(int argc, char** argv) {
    int n = 512;
    if (argc >= 2) {
        n = std::atoi(argv[1]);
        if (n <= 0) {
            std::cerr << "Invalid n; using n=512\n";
            n = 512;
        }
    }

    std::cout << "Floyd–Warshall CPU vs CUDA\n";
    std::cout << "n = " << n << " (matrix " << (long long)n * n << " entries)\n";

    std::vector<float> D0(n * n);
    init_random_graph(D0, n);

    // CPU run
    std::vector<float> D_cpu = D0;
    auto t0 = std::chrono::high_resolution_clock::now();
    floyd_warshall_cpu(D_cpu, n);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = elapsed_ms(t0, t1);
    std::cout << "CPU time:  " << cpu_ms << " ms\n";

    // CUDA run
    std::vector<float> D_gpu = D0;
    auto t2 = std::chrono::high_resolution_clock::now();
    floyd_warshall_cuda(D_gpu, n);
    auto t3 = std::chrono::high_resolution_clock::now();
    double gpu_ms = elapsed_ms(t2, t3);
    std::cout << "GPU time:  " << gpu_ms << " ms\n";

    // Verify
    float diff = max_abs_diff(D_cpu, D_gpu);
    std::cout << "Max abs diff CPU vs GPU: " << diff << "\n";
    if (gpu_ms > 0.0) {
        std::cout << "Speedup (CPU / GPU): " << cpu_ms / gpu_ms << "x\n";
    }

    return 0;
}
