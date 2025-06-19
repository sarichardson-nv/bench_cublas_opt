


#include <benchmark/benchmark.h>
#include <thrust/device_vector.h>

#include "bench_config.cuh"
#include "tiny_batched_gemm_no_coalesc.cuh"


inline constexpr auto tiny_batched_gemm_nocoal_3x3_rm = tiny_batched_gemm_nocoal<float, 3, 256, Eigen::RowMajor>;
inline constexpr auto tiny_batched_gemm_nocoal_3x3_cm = tiny_batched_gemm_nocoal<float, 3, 256, Eigen::ColMajor>;

static void bench_tiny_batched_gemm_nocoal_3x3_rm(benchmark::State &state) {
    constexpr int dim = 3;
    constexpr int size = dim * dim;
    const auto n_matrices = static_cast<int>(state.range(0));


    thrust::device_vector<float> a(size * n_matrices);
    thrust::device_vector<float> b(size * n_matrices);
    thrust::device_vector<float> c(size * n_matrices);

    for (auto _: state) {
        const auto threads = 256;
        const auto blocks = (n_matrices + threads - 1) / threads;

        float alpha = 1.0;
        float beta = 0.0;


        tiny_batched_gemm_nocoal_3x3_rm<<<blocks, threads>>>(
            raw_pointer_cast(a.data()),
            raw_pointer_cast(b.data()),
            raw_pointer_cast(c.data()),
            alpha,
            beta,
            n_matrices);


        cudaDeviceSynchronize();
    }
}

BENCHMARK(bench_tiny_batched_gemm_nocoal_3x3_rm)->Arg(kNumMatrices);


static void bench_tiny_batched_gemm_nocoal_3x3_cm(benchmark::State &state) {
    constexpr int dim = 3;
    constexpr int size = dim * dim;
    const auto n_matrices = static_cast<int>(state.range(0));


    thrust::device_vector<float> a(size * n_matrices);
    thrust::device_vector<float> b(size * n_matrices);
    thrust::device_vector<float> c(size * n_matrices);

    for (auto _: state) {
        const auto threads = 256;
        const auto blocks = (n_matrices + threads - 1) / threads;

        float alpha = 1.0;
        float beta = 0.0;


        tiny_batched_gemm_nocoal_3x3_cm<<<blocks, threads>>>(
            raw_pointer_cast(a.data()),
            raw_pointer_cast(b.data()),
            raw_pointer_cast(c.data()),
            alpha,
            beta,
            n_matrices);


        cudaDeviceSynchronize();
    }
}

BENCHMARK(bench_tiny_batched_gemm_nocoal_3x3_cm)->Arg(kNumMatrices);
