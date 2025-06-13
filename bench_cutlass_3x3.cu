//
//

#include <benchmark/benchmark.h>
#include <cutlass/gemm/device/gemm_batched.h>
#include <thrust/device_vector.h>

#include "bench_config.cuh"

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
