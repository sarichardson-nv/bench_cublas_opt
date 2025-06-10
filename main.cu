#include <cublas_v2.h>

#include <benchmark/benchmark.h>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/cub.cuh>
#include <Eigen/Core>
#include <thrust/device_vector.h>
#include <thrust/system/cuda/vector.h>

#ifdef USE_CUTLASS
#include <cutlass/gemm/device/gemm_batched.h>
#endif


static constexpr int64_t kNumMatrices = 1024 << 5;


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

BENCHMARK(bench_tiny_batched_gemm_3x3_rm)->Arg(kNumMatrices);

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

BENCHMARK(bench_tiny_batched_gemm_3x3_cm)->Arg(kNumMatrices);


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

BENCHMARK(bench_tiny_batched_gemm_3x3_rm_cls)->Arg(kNumMatrices);




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

BENCHMARK(bench_cublas_3x3_rm)->Arg(kNumMatrices);

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

BENCHMARK(bench_cublas_3x3_cm)->Arg(kNumMatrices);





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

BENCHMARK(bench_cutlass_3x3_rm)->Arg(kNumMatrices);



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

BENCHMARK(bench_cutlass_3x3_rm)->Arg(kNumMatrices);
#endif


BENCHMARK_MAIN();
