#include <cublas_v2.h>

#include <benchmark/benchmark.h>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <Eigen/Core>
#include <thrust/device_vector.h>


template<typename Sca, int MatrixDim, int BlockSize = 256, int Flags=Eigen::RowMajor>
__global__ void small_gemm(Sca *__restrict__ a, Sca *__restrict__ b, Sca *__restrict__ c, float alpha, float beta,
                           unsigned n_matrices) {
    constexpr int matrix_size = MatrixDim * MatrixDim;
    constexpr int block_stride = BlockSize * matrix_size;


    using Matrix = Eigen::Matrix<Sca, MatrixDim, MatrixDim, Flags>;

    using Load = cub::BlockLoad<Sca, BlockSize, matrix_size, cub::BLOCK_LOAD_DIRECT>;
    using Store = cub::BlockStore<Sca, BlockSize, matrix_size, cub::BLOCK_STORE_DIRECT>;

    using SMem =union {
        typename Load::TempStorage load_tmp;
        typename Store::TempStorage store_tmp;
    };

    // ReSharper disable once CppTooWideScope
    __shared__ SMem shared_mem;

    Matrix A;
    Matrix B;
    Matrix C;

    using ThreadArray = Sca[matrix_size];

    const auto n_blocks = (n_matrices + BlockSize - 1) / BlockSize;

    for (unsigned block_i = blockIdx.x; block_i < n_blocks; block_i += gridDim.x) {
        Load(shared_mem.load_tmp).Load(a + block_i * block_stride, *reinterpret_cast<ThreadArray*>(A.data()));
        __syncthreads();
        Load(shared_mem.load_tmp).Load(b + block_i * block_stride, *reinterpret_cast<ThreadArray*>(B.data()));
        __syncthreads();
        Load(shared_mem.load_tmp).Load(c + block_i * block_stride, *reinterpret_cast<ThreadArray*>(C.data()));

        C = alpha * A * B + beta * C;

        Store(shared_mem.store_tmp).Store(c + block_i * block_stride, *reinterpret_cast<ThreadArray*>(C.data()));
    }

}

inline constexpr auto small_gemm_3x3_rm = small_gemm<float, 3, 256, Eigen::RowMajor>;
inline constexpr auto small_gemm_3x3_cm = small_gemm<float, 3, 256, Eigen::ColMajor>;

static void bench_small_gemm_3x3_rm(benchmark::State &state) {
    constexpr int dim = 3;
    constexpr int size = 9;
    const auto n_matrices = static_cast<int>(state.range(0));
    using Index = int64_t;


    thrust::device_vector<float> a(size * n_matrices);
    thrust::device_vector<float> b(size * n_matrices);
    thrust::device_vector<float> c(size * n_matrices);

    for (auto _: state) {

        const auto threads = 256;
        const auto blocks = (n_matrices + threads - 1) / threads;

        float alpha = 1.0;
        float beta = 0.0;


        small_gemm_3x3_rm<<<blocks, threads>>>(
            raw_pointer_cast(a.data()),
            raw_pointer_cast(b.data()),
            raw_pointer_cast(c.data()),
            alpha,
            beta,
            n_matrices);


        cudaDeviceSynchronize();
    }

}

BENCHMARK(bench_small_gemm_3x3_rm)->Arg(1024 << 5);

static void bench_small_gemm_3x3_cm(benchmark::State &state) {
    constexpr int dim = 3;
    constexpr int size = 9;
    const auto n_matrices = static_cast<int>(state.range(0));
    using Index = int64_t;


    thrust::device_vector<float> a(size * n_matrices);
    thrust::device_vector<float> b(size * n_matrices);
    thrust::device_vector<float> c(size * n_matrices);

    for (auto _: state) {

        const auto threads = 256;
        const auto blocks = (n_matrices + threads - 1) / threads;

        float alpha = 1.0;
        float beta = 0.0;


        small_gemm_3x3_cm<<<blocks, threads>>>(
            raw_pointer_cast(a.data()),
            raw_pointer_cast(b.data()),
            raw_pointer_cast(c.data()),
            alpha,
            beta,
            n_matrices);


        cudaDeviceSynchronize();
    }

}

BENCHMARK(bench_small_gemm_3x3_cm)->Arg(1024 << 5);

static void bench_cublas_3x3_rm(benchmark::State &state) {
    constexpr int dim = 3;
    constexpr int size = 9;
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
            CUBLAS_OP_T, CUBLAS_OP_T,
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

BENCHMARK(bench_cublas_3x3_rm)->Arg(1024 << 5);

static void bench_cublas_3x3_cm(benchmark::State &state) {
    constexpr int dim = 3;
    constexpr int size = 9;
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

BENCHMARK(bench_cublas_3x3_cm)->Arg(1024 << 5);


BENCHMARK_MAIN();
