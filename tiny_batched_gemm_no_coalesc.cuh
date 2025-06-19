//
//

#ifndef TINY_BATCHED_GEMM_CUH
#define TINY_BATCHED_GEMM_CUH

#include <Eigen/Core>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>


template<typename Sca, int MatrixDim, int BlockSize = 256, int Flags = Eigen::RowMajor>
__global__ void tiny_batched_gemm_nocoal(Sca *__restrict__ a, Sca *__restrict__ b, Sca *__restrict__ c, float alpha,
                                  float beta,
                                  unsigned n_matrices) {
    constexpr int matrix_size = MatrixDim * MatrixDim;
    constexpr int block_stride = BlockSize * matrix_size;


    using Matrix = Eigen::Matrix<Sca, MatrixDim, MatrixDim, Flags>;
    using Map = Eigen::Map<Matrix>;
    using ConstMap = Eigen::Map<const Matrix>;

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
        // Load(shared_mem.load_tmp).Load(a + block_i * block_stride, *reinterpret_cast<ThreadArray *>(A.data()));
        // __syncthreads();
        // Load(shared_mem.load_tmp).Load(b + block_i * block_stride, *reinterpret_cast<ThreadArray *>(B.data()));
        // __syncthreads();
        // Load(shared_mem.load_tmp).Load(c + block_i * block_stride, *reinterpret_cast<ThreadArray *>(C.data()));
        // __syncthreads();
        A = ConstMap(a + block_i * block_stride);
        B = ConstMap(b + block_i * block_stride);
        C = ConstMap(c + block_i * block_stride);

        C = alpha * A * B + beta * C;

        Map(c + block_i * block_stride) = C;
        // Store(shared_mem.store_tmp).Store(c + block_i * block_stride, *reinterpret_cast<ThreadArray *>(C.data()));
        // __syncthreads();
    }
}



#endif //TINY_BATCHED_GEMM_CUH
