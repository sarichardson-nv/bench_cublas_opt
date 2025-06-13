//
//

#ifndef TINY_BATCHED_GEMM_CLS_CUH
#define TINY_BATCHED_GEMM_CLS_CUH

#include <Eigen/Core>

#include "vectorized_copy.cuh"

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

#endif //TINY_BATCHED_GEMM_CLS_CUH
