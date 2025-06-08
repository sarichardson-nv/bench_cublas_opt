#include <cublas_v2.h>

#include <benchmark/benchmark.h>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/warp/warp_load.cuh>
#include <cub/warp/warp_store.cuh>
#include <Eigen/Core>
#include <thrust/device_vector.h>


template<typename Sca, int MatrixDim, int BlockSize = 256, int Flags = Eigen::RowMajor>
__global__ void tiny_batched_gemm(Sca *__restrict__ a, Sca *__restrict__ b, Sca *__restrict__ c, float alpha,
                                  float beta,
                                  unsigned n_matrices) {
    constexpr int matrix_size = MatrixDim * MatrixDim;
    constexpr int block_stride = BlockSize * matrix_size;


    using Matrix = Eigen::Matrix<Sca, MatrixDim, MatrixDim, Flags>;

    using Load = cub::BlockLoad<Sca, BlockSize, matrix_size, cub::BLOCK_LOAD_TRANSPOSE>;
    using Store = cub::BlockStore<Sca, BlockSize, matrix_size, cub::BLOCK_STORE_TRANSPOSE>;

    using SMem = union {
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
        Load(shared_mem.load_tmp).Load(a + block_i * block_stride, *reinterpret_cast<ThreadArray *>(A.data()));
        __syncthreads();
        Load(shared_mem.load_tmp).Load(b + block_i * block_stride, *reinterpret_cast<ThreadArray *>(B.data()));
        __syncthreads();
        Load(shared_mem.load_tmp).Load(c + block_i * block_stride, *reinterpret_cast<ThreadArray *>(C.data()));
        __syncthreads();

        C = alpha * A * B + beta * C;

        Store(shared_mem.store_tmp).Store(c + block_i * block_stride, *reinterpret_cast<ThreadArray *>(C.data()));
        __syncthreads();
    }
}

inline constexpr auto tiny_batched_gemm_3x3_rm = tiny_batched_gemm<float, 3, 256, Eigen::RowMajor>;
inline constexpr auto tiny_batched_gemm_3x3_cm = tiny_batched_gemm<float, 3, 256, Eigen::ColMajor>;

static void bench_tiny_batched_gemm_3x3_rm(benchmark::State &state) {
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


        tiny_batched_gemm_3x3_rm<<<blocks, threads>>>(
            raw_pointer_cast(a.data()),
            raw_pointer_cast(b.data()),
            raw_pointer_cast(c.data()),
            alpha,
            beta,
            n_matrices);


        cudaDeviceSynchronize();
    }
}

BENCHMARK(bench_tiny_batched_gemm_3x3_rm)->Arg(1024 << 5);

static void bench_tiny_batched_gemm_3x3_cm(benchmark::State &state) {
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


        tiny_batched_gemm_3x3_cm<<<blocks, threads>>>(
            raw_pointer_cast(a.data()),
            raw_pointer_cast(b.data()),
            raw_pointer_cast(c.data()),
            alpha,
            beta,
            n_matrices);


        cudaDeviceSynchronize();
    }
}

BENCHMARK(bench_tiny_batched_gemm_3x3_cm)->Arg(1024 << 5);

/**
 * @brief compute batched gemm operations on small matrices cooperatively
 *
 * This is Mike Giles' algorithm for computing batched products of matrices using
 * cooperative thread groups. His original code was written in CUDA-C, so I've
 * translated it to C++ and templates and some other C++ niceties to simplify
 * the code.
 */
template<typename Sca, int MatrixDim, int NThreadsPerDim, int BlockSize=256, int Flags = Eigen::RowMajor>
__global__ void small_batched_cooperative_gemm(Sca *__restrict__ a, Sca *__restrict__ b, Sca *__restrict__ c,
                                               float alpha, float beta,
                                               unsigned n_matrices) {
    static_assert(NThreadsPerDim > 0 && NThreadsPerDim < 32 && (NThreadsPerDim & (NThreadsPerDim - 1)) == 0,
        "NThreadsPerDim must be a power of 2 and < 32"
        );

    constexpr int full_matrix_dim = MatrixDim;
    constexpr int full_matrix_size = MatrixDim * MatrixDim;

    constexpr int matrix_dim = (MatrixDim + NThreadsPerDim - 1) / NThreadsPerDim;
    constexpr int matrix_size = matrix_dim * matrix_dim;
    constexpr int threads_per_dim = NThreadsPerDim;
    constexpr int threads_per_matrix = NThreadsPerDim * NThreadsPerDim;
    constexpr int matrices_per_block = BlockSize / threads_per_matrix;
    constexpr auto pass_stride = full_matrix_size * matrices_per_block;

    constexpr unsigned thread_mask = 0xFF'FF'FF'FF;

    /*
     * This is a tile of a larger matrix that exists virtually amongst a group of
     * threads_per_matrix threads within a single warp. These matrix tiles are
     * used cooperatively to compute the final result. The layout is more
     * complicated here because the loads for each thread are no longer
     * contiguous, so we can't us the BlockLoad primitive operation. For
     * clarity, suppose we have a 4x4 matrix in row-major order, and we use 2 tiles per dim.
     * The picture is as follows.
     *
     *    +---+---+
     *    | 0 | 1 |
     *    +---+---+
     *    | 2 | 3 |
     *    +---+---+
     *
     * The numbers above refer to the lane id. In terms of access pattern,
     * this translates to the following picture. The top row shows the
     * regions of contiguous data accessed by each thread (indexed by lane id)
     * and the bottom row shows the corresponding rows as arranged in
     * memory.
     *
     *   |   0   |   1   |   0   |   1   |   2   |   3   |   2   |   3   |
     *   |     row 0     |     row 1     |     row 2     |     row 3     |
     *
     */
    using Matrix = Eigen::Matrix<Sca, matrix_dim, matrix_dim, Flags>;

    using Load = cub::WarpLoad<Sca, matrix_size, cub::WARP_LOAD_TRANSPOSE, threads_per_matrix>;


    const auto lane = threadIdx.x % threads_per_matrix;
    const auto block_row = lane / NThreadsPerDim;
    const auto block_col = lane % NThreadsPerDim;

    Matrix A_tile;
    Matrix B_tile;
    Matrix C_tile;

    const auto n_passes = (n_matrices + matrices_per_block - 1) / matrices_per_block;

    for (unsigned pass_idx = blockIdx.x; pass_idx < n_passes; pass_idx += matrices_per_block) {

        const auto* start_of_a_block = a + pass_idx * pass_stride;
        const auto* start_of_b_block = b + pass_idx * pass_stride;
        auto* start_of_c_block = c + pass_idx * pass_stride;

        auto lane1 = block_col + block_col * threads_per_dim;
        auto lane2 = (lane + 1) % threads_per_matrix + threads_per_matrix * block_row;

        const auto* start_of_a_tile = start_of_a_block + block_row * matrix_dim + block_col;
        const auto* start_of_b_tile = start_of_b_block + block_row * matrix_dim + block_col;
        auto* start_of_c_tile = start_of_c_block + block_row * matrix_dim + block_col;

        C_tile.setZero();


        for (unsigned tile_idx=0; tile_idx < threads_per_dim; tile_idx++) {

            for (unsigned j=0; j<matrix_dim; ++j) {
                for (unsigned k=0; k<matrix_dim; ++k) {
                    auto rhs_val = __shfl_sync(thread_mask, B_tile(k, j), lane1, threads_per_matrix);

                    for (unsigned i=0; i<matrix_dim; ++i) {
                        C_tile += A_tile(i, k)*rhs_val;
                    }
                }
            }

            lane1 = (lane1 + threads_per_dim) % threads_per_matrix;

            for (unsigned j=0; j<matrix_dim; ++j) {
                for (unsigned i=0; i<matrix_dim; ++i) {
                    A_tile(i, j) = __shfl_sync(thread_mask, A_tile(i, j), lane2, threads_per_matrix);
                }
            }

        } // end of block loop

        /*
         * Now C_tile contains the result of A*B, so we need to
         * accumulate compute beta*C + C_tile and write the result
         * back to memory. We're done with A_tile and B_tile so we can
         * use this reuse these to store the temporary results. No
         * cooperative action is needed here since addition and
         * scalar multiplication are coordinate-wise.
         */

        // read C into A_tile

        B_tile.noalias() = beta*A_tile + alpha*C_tile;


        // write B_tile back to C

    }


}


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
