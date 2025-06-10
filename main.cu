#include <cublas_v2.h>

#include <benchmark/benchmark.h>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/cub.cuh>
#include <cub/warp/warp_load.cuh>
#include <cub/warp/warp_store.cuh>
#include <Eigen/Core>
#include <thrust/device_vector.h>
#include <thrust/system/cuda/vector.h>

#ifdef USE_CUTLASS
#include <cutlass/gemm/device/gemm_batched.h>
#endif


template <typename T, unsigned MatrixDim, unsigned BlockSize>
__device__ __forceinline__
void vectorized_copy(T* __restrict dst_ptr, const T* __restrict src_ptr, unsigned n_matrices) {

    constexpr auto vector_size = 16 / sizeof(T); // vectors are 128 bits
    constexpr auto matrix_size = MatrixDim * MatrixDim;
    constexpr auto block_size = BlockSize;
    constexpr auto smem_size = BlockSize * matrix_size;

    // I'm just assuming that smem_size is an exact multiple of vector_size
    const auto n_vector_loads = n_matrices / vector_size;

    using Vector = cub::CubVector<T, vector_size>;

    auto* in_ptr = reinterpret_cast<const Vector *>(src_ptr);
    auto* out_ptr = reinterpret_cast<Vector *>(dst_ptr);

    for (unsigned i=threadIdx.x; i<n_vector_loads; i+=blockDim.x) {
        out_ptr[i] = in_ptr[i];
    }
}






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


template<typename Sca, int MatrixDim, int BlockSize = 256, int Flags = Eigen::RowMajor>
__global__ void tiny_batched_gemm_cls(Sca *__restrict__ a, Sca *__restrict__ b, Sca *__restrict__ c, float alpha,
                                  float beta,
                                  unsigned n_matrices) {
    constexpr int matrix_dim = MatrixDim;
    constexpr int matrix_size = MatrixDim * MatrixDim;
    constexpr int block_stride = BlockSize * matrix_size;


    using Matrix = Eigen::Matrix<Sca, MatrixDim, MatrixDim, Flags>;


    // ReSharper disable once CppTooWideScope
    __shared__ Sca shared_mem[BlockSize * matrix_size];

    Matrix A;
    Matrix B;
    Matrix C;

    auto load = [&](Matrix& dst, const Sca* ptr) {
        vectorized_copy<float, MatrixDim, BlockSize>(shared_mem, ptr, BlockSize);

        __syncthreads();
        for (int i=0; i<MatrixDim; ++i) {
            for (int j=0; j<MatrixDim; ++j){ 
                dst(i, j) = shared_mem[threadIdx.x * matrix_size + i * matrix_dim + j];
            }
        }
        __syncthreads();
    };

    auto store = [&](const Matrix& src, Sca* dst) {
        for (int i=0; i<MatrixDim; ++i) {
            for (int j=0; j<MatrixDim; ++j){ 
                shared_mem[threadIdx.x * matrix_size + i * matrix_dim + j] = src(i, j);
            }
        }
        __syncthreads();

        vectorized_copy<float, MatrixDim, BlockSize>(dst, shared_mem, BlockSize);
        __syncthreads();
    };

    using ThreadArray = Sca[matrix_size];

    const auto n_blocks = (n_matrices + BlockSize - 1) / BlockSize;

    for (unsigned block_i = blockIdx.x; block_i < n_blocks; block_i += gridDim.x) {
        load(A, a + block_i * block_stride);
        load(B, b + block_i * block_stride);
        load(C, c + block_i * block_stride);

        C = alpha * A * B + beta * C;

        store(C, c + block_i * block_stride);
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


inline constexpr auto tiny_batched_gemm_3x3_rm_cls = tiny_batched_gemm_cls<float, 3, 256, Eigen::RowMajor>;

static void bench_tiny_batched_gemm_3x3_rm_cls(benchmark::State &state) {
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


        tiny_batched_gemm_3x3_rm_cls<<<blocks, threads>>>(
            raw_pointer_cast(a.data()),
            raw_pointer_cast(b.data()),
            raw_pointer_cast(c.data()),
            alpha,
            beta,
            n_matrices);


        cudaDeviceSynchronize();
    }
}

BENCHMARK(bench_tiny_batched_gemm_3x3_rm_cls)->Arg(1024 << 5);

static void bench_tiny_batched_gemm_4x4_rm(benchmark::State &state) {
    constexpr int dim = 4;
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

BENCHMARK(bench_tiny_batched_gemm_4x4_rm)->Arg(1024 << 5);





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

    constexpr auto shared_memory_size = matrices_per_block * full_matrix_size;

    __shared__ Sca shared_mem[shared_memory_size];

    const auto lane = threadIdx.x % threads_per_matrix;
    const auto warp_id = threadIdx.x / threads_per_matrix;
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

        /*
         * We have to lead the data into the matrix tiles A_tile and B_tile.
         *
         * This is a two-stage process, first loading data into shared memory and then
         * performing a second load into thread-local data. This is so that the first
         * stage can benefit from coalesced and vectorized loads from global memory,
         * improving the memory bandwidth. We've written a little function that
         * performs these vectorized, coalesced loads, although this needs some work
         * to address the more general case. The remaining part is to take the tile
         * data from each matrix in to the temporary matrices. To keep this simple,
         * we assume the tiles are arranged using the same ordering as the matrix itself.
         * This means we load matrix_size elements from shared memory at offset given
         * by both the warp_id, and the block indices bi and bj.
         */
        do {
            vectorized_copy<Sca, MatrixDim, BlockSize>(shared_mem, start_of_a_block, matrices_per_block);
            __syncthreads();

            for (int i=0; i<matrix_dim; ++i) {
                for (int j=0; j<matrix_dim; ++j) {
                    A_tile(i, j) = shared_mem[
                        warp_id * full_matrix_size + full_matrix_dim * (block_row * matrix_dim + i) + block_col * matrix_dim + j
                        ];
                }
            }
            __syncthreads();
        } while (false);
        do {
            vectorized_copy<Sca, MatrixDim, BlockSize>(shared_mem, start_of_b_block, matrices_per_block);
            __syncthreads();

            for (int i=0; i<matrix_dim; ++i) {
                for (int j=0; j<matrix_dim; ++j) {
                    B_tile(i, j) = shared_mem[
                        warp_id * full_matrix_size + full_matrix_dim * (block_row * matrix_dim + i) + block_col * matrix_dim + j
                        ];
                }
            }
            __syncthreads();
        } while (false);


        for (unsigned tile_idx=0; tile_idx < threads_per_dim; tile_idx++) {

            for (unsigned j=0; j<matrix_dim; ++j) {
                for (unsigned k=0; k<matrix_dim; ++k) {
                    auto rhs_val = __shfl_sync(thread_mask, B_tile(k, j), lane1, threads_per_matrix);

                    for (unsigned i=0; i<matrix_dim; ++i) {
                        C_tile(i, j) += A_tile(i, k)*rhs_val;
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
        do {
            vectorized_copy<Sca, MatrixDim, BlockSize>(shared_mem, start_of_c_block, matrices_per_block);
            __syncthreads();

            for (int i=0; i<matrix_dim; ++i) {
                for (int j=0; j<matrix_dim; ++j) {
                    A_tile(i, j) = shared_mem[
                        warp_id * full_matrix_size + full_matrix_dim * (block_row * matrix_dim + i) + block_col * matrix_dim + j
                        ];
                }
            }
            __syncthreads();
        } while (false);

        B_tile.noalias() = beta*A_tile + alpha*C_tile;

        // write B_tile back to C
        do {
            for (int i=0; i<matrix_dim; ++i) {
                for (int j=0; j<matrix_dim; ++j) {
                    shared_mem[
                        warp_id * full_matrix_size + full_matrix_dim * (block_row * matrix_dim + i) + block_col * matrix_dim + j
                        ] = B_tile(i, j);
                }
            }

            __syncthreads();
            vectorized_copy<Sca, MatrixDim, BlockSize>(start_of_c_block, shared_mem, matrices_per_block);
        } while (false);


    }
}



inline constexpr auto small_batched_cooperative_gemm_4x4_rm = small_batched_cooperative_gemm<float, 4, 2, 256, Eigen::RowMajor>;
inline constexpr auto small_batched_cooperative_gemm_4x4_cm = small_batched_cooperative_gemm<float, 4, 2, 256, Eigen::ColMajor>;

static void bench_small_batched_cooperative_gemm_4x4_rm(benchmark::State &state) {
    constexpr int dim = 4;
    constexpr int size = 16;
    auto n_matrices = static_cast<int>(state.range(0));


    thrust::device_vector<float> a(size * n_matrices, 1.0f);
    thrust::device_vector<float> b(size * n_matrices, 1.0f);
    thrust::device_vector<float> c(size * n_matrices, 0.0f);


    for (auto _: state) {
        const auto threads = 256;
        const auto blocks = (n_matrices + threads - 1) / threads;

        float alpha = 1.0;
        float beta = 0.0;

        small_batched_cooperative_gemm_4x4_rm<<<blocks, threads>>>(
            raw_pointer_cast(a.data()),
            raw_pointer_cast(b.data()),
            raw_pointer_cast(c.data()),
            alpha,
            beta,
            n_matrices);


        cudaDeviceSynchronize();
    }

}

BENCHMARK(bench_small_batched_cooperative_gemm_4x4_rm)->Arg(1024<<5);

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





#ifdef USE_CUTLASS

static void bench_cutlass_3x3_rm(benchmark::State &state) {

    // CUTLASS GEMM Configuration (You can adjust these parameters)
    using CutlassBatchedGemm = cutlass::gemm::device::GemmBatched<
        float,                             // Element type for A matrix
        cutlass::layout::RowMajor,         // Layout of A matrix
        float,                             // Element type for B matrix
        cutlass::layout::RowMajor,         // Layout of B matrix
        float,                             // Element type for C/D matrix
        cutlass::layout::RowMajor>;         // Layout of C/D matrix


   constexpr int matrixDim = 3; // Matrix dimensions: 3x3
    constexpr int matrix_size = matrixDim * matrixDim;
    const int n_matrices = static_cast<int>(state.range(0));

    // Allocate data for A, B, and C matrices
    thrust::device_vector<float> A(matrix_size * n_matrices, 1.0f); // Initialize all to 1.0
    thrust::device_vector<float> B(matrix_size * n_matrices, 1.0f); // Initialize all to 1.0
    thrust::device_vector<float> C(matrix_size * n_matrices, 0.0f); // Initialize all to 0.0

    const float alpha = 1.0f; // Scalar multiplier for A * B
    const float beta = 0.0f;  // Scalar multiplier for C

    // CUTLASS Batched GEMM arguments
    typename CutlassBatchedGemm::Arguments arguments(
        {matrixDim, matrixDim, matrixDim},  // Problem size (M, N, K)
        {raw_pointer_cast(A.data()), matrixDim}, // Pointer and leading dimension of A
         matrix_size,
        {raw_pointer_cast(B.data()), matrixDim}, // Pointer and leading dimension of B
         matrix_size,
        {raw_pointer_cast(C.data()), matrixDim}, // Pointer and leading dimension of C
         matrix_size,
        {raw_pointer_cast(C.data()), matrixDim}, // Pointer and leading dimension of D
         matrix_size,
        {alpha, beta},                           // Scalars alpha and beta
        n_matrices                               // Number of matrices (batch size)
    );

    // Create an instance of Batched GEMM
    CutlassBatchedGemm batched_gemm_op;

    // Check if the given configuration is supported
    cutlass::Status status = batched_gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS BatchedGemm configuration is not supported!");
    }

    for (auto _ : state) {
        // Launch the CUTLASS Batched GEMM operation
        status = batched_gemm_op(arguments);
        if (status != cutlass::Status::kSuccess) {
            throw std::runtime_error("CUTLASS BatchedGemm execution failed!");
        }

        // Ensure computation completes before proceeding
        // cudaDeviceSynchronize();
    }

}

BENCHMARK(bench_cutlass_3x3_rm)->Arg(1024 << 5);



static void bench_cutlass_3x3_cm(benchmark::State &state) {

    // CUTLASS GEMM Configuration (You can adjust these parameters)
    using CutlassBatchedGemm = cutlass::gemm::device::GemmBatched<
        float,                             // Element type for A matrix
        cutlass::layout::ColumnMajor,         // Layout of A matrix
        float,                             // Element type for B matrix
        cutlass::layout::ColumnMajor,         // Layout of B matrix
        float,                             // Element type for C/D matrix
        cutlass::layout::ColumnMajor>;         // Layout of C/D matrix


   constexpr int matrixDim = 3; // Matrix dimensions: 3x3
    constexpr int matrix_size = matrixDim * matrixDim;
    const int n_matrices = static_cast<int>(state.range(0));

    // Allocate data for A, B, and C matrices
    thrust::device_vector<float> A(matrix_size * n_matrices, 1.0f); // Initialize all to 1.0
    thrust::device_vector<float> B(matrix_size * n_matrices, 1.0f); // Initialize all to 1.0
    thrust::device_vector<float> C(matrix_size * n_matrices, 0.0f); // Initialize all to 0.0

    const float alpha = 1.0f; // Scalar multiplier for A * B
    const float beta = 0.0f;  // Scalar multiplier for C

    // CUTLASS Batched GEMM arguments
    typename CutlassBatchedGemm::Arguments arguments(
        {matrixDim, matrixDim, matrixDim},  // Problem size (M, N, K)
        {raw_pointer_cast(A.data()), matrixDim}, // Pointer and leading dimension of A
         matrix_size,
        {raw_pointer_cast(B.data()), matrixDim}, // Pointer and leading dimension of B
         matrix_size,
        {raw_pointer_cast(C.data()), matrixDim}, // Pointer and leading dimension of C
         matrix_size,
        {raw_pointer_cast(C.data()), matrixDim}, // Pointer and leading dimension of D
         matrix_size,
        {alpha, beta},                           // Scalars alpha and beta
        n_matrices                               // Number of matrices (batch size)
    );

    // Create an instance of Batched GEMM
    CutlassBatchedGemm batched_gemm_op;

    // Check if the given configuration is supported
    cutlass::Status status = batched_gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS BatchedGemm configuration is not supported!");
    }

    for (auto _ : state) {
        // Launch the CUTLASS Batched GEMM operation
        status = batched_gemm_op(arguments);
        if (status != cutlass::Status::kSuccess) {
            throw std::runtime_error("CUTLASS BatchedGemm execution failed!");
        }

        // Ensure computation completes before proceeding
        // cudaDeviceSynchronize();
    }

}

BENCHMARK(bench_cutlass_3x3_rm)->Arg(1024 << 5);
#endif


BENCHMARK_MAIN();
