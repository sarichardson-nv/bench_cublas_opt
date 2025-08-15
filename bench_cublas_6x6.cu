//
//


#include <cublas_v2.h>

#include <benchmark/benchmark.h>
#include <thrust/device_vector.h>


#include "bench_config.cuh"


// static void bench_cublas_6x6_rm(benchmark::State &state) {
//     constexpr int dim = 6;
//     constexpr int size = 36;
//     auto n_matrices = static_cast<int>(state.range(0));
//     using Index = int;
//
//     thrust::device_vector<float> a(size * n_matrices);
//     thrust::device_vector<float> b(size * n_matrices);
//     thrust::device_vector<float> c(size * n_matrices);
//
//     cublasHandle_t handle;
//     cublasCreate(&handle);
//
//     for (auto _: state) {
//         Index m = dim;
//         Index n = dim;
//         Index k = dim;
//         Index strideA = size;
//         Index strideB = size;
//         Index strideC = size; // The result is the same for every computation
//         Index batchCount = n_matrices;
//
//         Index lda = dim;
//         Index ldb = dim;
//         Index ldc = dim;
//
//         float alpha = 1.0;
//         float beta = 0.0;
//
//         const auto *A = raw_pointer_cast(a.data());
//         const auto *B = raw_pointer_cast(b.data());
//         auto *C = raw_pointer_cast(c.data());
//
//         auto result = cublasSgemmStridedBatched(
//             handle,
//             CUBLAS_OP_T, CUBLAS_OP_T,
//             m, n, k,
//             &alpha,
//             A, lda,
//             strideA,
//             B, ldb,
//             strideB,
//             &beta,
//             C, ldc,
//             strideC,
//             batchCount
//         );
//
//         benchmark::DoNotOptimize(result);
//     }
//
//
//     cublasDestroy(handle);
// }
//
// BENCHMARK(bench_cublas_6x6_rm)->Arg(kNumMatrices);

static void bench_cublas_6x6_cm(benchmark::State &state) {
    constexpr int dim = 6;
    constexpr int size = 36;
    auto n_matrices = static_cast<int>(state.range(0));
    using Index = int;

    thrust::device_vector<float> a(size * n_matrices);
    thrust::device_vector<float> b(size * n_matrices);
    thrust::device_vector<float> c(size * n_matrices);

    cublasHandle_t handle;
    cublasCreate(&handle);

    for (auto _: state) {
        Index m = dim;
        Index n = dim;
        Index k = dim;
        Index strideA = size;
        Index strideB = size;
        Index strideC = size; // The result is the same for every computation
        Index batchCount = n_matrices;

        Index lda = dim;
        Index ldb = dim;
        Index ldc = dim;

        float alpha = 1.0;
        float beta = 0.0;

        const auto *A = raw_pointer_cast(a.data());
        const auto *B = raw_pointer_cast(b.data());
        auto *C = raw_pointer_cast(c.data());

        auto result = cublasSgemmStridedBatched(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            m, n, k,
            &alpha,
            A, lda,
            strideA,
            B, ldb,
            strideB,
            &beta,
            C, ldc,
            strideC,
            batchCount
        );

        benchmark::DoNotOptimize(result);
    }


    cublasDestroy(handle);
}

BENCHMARK(bench_cublas_6x6_cm)->Arg(kNumMatrices);
