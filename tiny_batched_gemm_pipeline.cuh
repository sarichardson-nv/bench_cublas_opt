//
//

#ifndef TINY_BATCHED_GEMM_CUH
#define TINY_BATCHED_GEMM_CUH

#include <Eigen/Core>

#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <cuda/std/functional>


#pragma nv_suppress static_var_with_dynamic_init


template<typename Scalar_, int MatrixDim, int BlockSize, int Flags>
__forceinline__ __device__
void compute_tiny_batched_gemm(Scalar_ *__restrict a, Scalar_ *__restrict b, Scalar_ *__restrict c,
                               Scalar_ const &alpha, Scalar_ const &beta, unsigned remaining_matrices) {
    using Matrix = Eigen::Matrix<Scalar_, MatrixDim, MatrixDim, Flags>;

    constexpr auto matrix_size = MatrixDim * MatrixDim;
    Matrix A;
    Matrix B;
    Matrix C;

    for (int i = 0; i < MatrixDim; ++i) {
        for (int j = 0; j < MatrixDim; ++j) {
            B(i, j) = b[threadIdx.x * matrix_size + i * MatrixDim + j];
            C(i, j) = c[threadIdx.x * matrix_size + i * MatrixDim + j];
        }
    }

    C = A * B;

    for (int i = 0; i < MatrixDim; ++i) {
        for (int j = 0; j < MatrixDim; ++j) {
            b[threadIdx.x * matrix_size + i * MatrixDim + j] = alpha*C(i, j);
        }
    }

    for (unsigned i = threadIdx.x; i < remaining_matrices * matrix_size; i += BlockSize) {
        a[i] = alpha*a[i] + b[i];
    }
}

template<typename Sca, int MatrixDim, unsigned BlockSize = 256, int Flags = Eigen::RowMajor>
__global__ void tiny_batched_gemm_pipeline(Sca *__restrict__ a, Sca *__restrict__ b, Sca *__restrict__ c, float alpha,
                                  float beta,
                                  unsigned n_matrices) {
    constexpr int matrix_size = MatrixDim * MatrixDim;
    constexpr int block_stride = BlockSize * matrix_size;

    auto group = cooperative_groups::this_thread_block();
    constexpr auto scope = cuda::thread_scope_block;
    constexpr unsigned stages_count = 2;

    constexpr auto smem_bank_size = BlockSize * matrix_size;

    __shared__ alignas(16) Sca shared_mem[stages_count][2][smem_bank_size];
    __shared__ cuda::pipeline_shared_state<scope, stages_count> shared_state;

    auto pipeline = cuda::make_pipeline(group, &shared_state);


    const auto n_blocks = (n_matrices + BlockSize - 1) / BlockSize;

    for (unsigned block_i = blockIdx.x; block_i < n_blocks; block_i += BlockSize) {
        auto block_begin = block_i * BlockSize;
        auto remaining_matrices = cuda::std::min(BlockSize, n_matrices - block_begin);

        pipeline.producer_acquire();
        cuda::memcpy_async(group, &shared_mem[0][0][0], b, sizeof(Sca) * remaining_matrices * matrix_size, pipeline);
        cuda::memcpy_async(group, &shared_mem[0][1][0], c, sizeof(Sca) * remaining_matrices * matrix_size, pipeline);
        pipeline.producer_commit();

        for (unsigned stage_i = 1; stage_i < stages_count; ++stage_i) {
            pipeline.producer_acquire();
            auto next_block_begin = block_begin + BlockSize;
            auto next_remaining_matrices = cuda::std::min(BlockSize, n_matrices - next_block_begin);
            cuda::memcpy_async(group,
                               shared_mem[stage_i][0],
                               &b[next_block_begin * matrix_size],
                               sizeof(Sca) * next_remaining_matrices * matrix_size,
                               pipeline);
            cuda::memcpy_async(group,
                               shared_mem[stage_i][1],
                               &c[next_block_begin * matrix_size],
                               sizeof(Sca) * next_remaining_matrices * matrix_size,
                               pipeline);

            pipeline.producer_commit();

            pipeline.consumer_wait();

            auto stage_idx = stage_i - 1;
            compute_tiny_batched_gemm<Sca, MatrixDim, BlockSize, Flags>(
                &a[block_begin],
                shared_mem[stage_idx][1],
                shared_mem[stage_idx][0],
                alpha, beta, remaining_matrices);

            pipeline.consumer_release();
        }

        pipeline.consumer_wait();
        auto stage_idx = (stages_count - 1) % stages_count;
        compute_tiny_batched_gemm<Sca, MatrixDim, BlockSize, Flags>(
            &a[block_begin],
            shared_mem[stage_idx][1],
            shared_mem[stage_idx][0],
            alpha, beta, remaining_matrices);
        pipeline.consumer_release();
    }
}


#endif //TINY_BATCHED_GEMM_CUH
