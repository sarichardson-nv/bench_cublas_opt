//
//

#ifndef TINY_BATCHED_GEMM_CUH
#define TINY_BATCHED_GEMM_CUH

#include <Eigen/Core>

#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <cuda/std/functional>


#pragma nv_suppress static_var_with_dynamic_init



template<typename Sca, int MatrixDim, unsigned BlockSize = 256, int Flags = Eigen::RowMajor>
__global__ void tiny_batched_gemm_pipeline(Sca *__restrict__ a, Sca *__restrict__ b, Sca *__restrict__ c, float alpha,
                                           float beta,
                                           unsigned n_matrices) {
    constexpr int matrix_dim = MatrixDim;
    constexpr int matrix_size = MatrixDim * MatrixDim;
    constexpr int block_stride = BlockSize * matrix_size;
    constexpr int matrices_per_block = BlockSize;

    using Matrix = Eigen::Matrix<Sca, MatrixDim, MatrixDim, Flags>;

    auto group = cooperative_groups::this_thread_block();
    constexpr auto scope = cuda::thread_scope_block;
    constexpr unsigned stages_count = 2;

    constexpr auto smem_bank_size = BlockSize * matrix_size;

    __shared__ alignas(16) Sca shared_mem[2][smem_bank_size];
    __shared__ cuda::pipeline_shared_state<scope, stages_count> shared_state;

    auto pipeline = cuda::make_pipeline(group, &shared_state);


    const auto n_blocks = (n_matrices + BlockSize - 1) / BlockSize;


    Matrix A_tile, B_tile, C_tile;

    int seq = 0;
    for (unsigned block_i = blockIdx.x; block_i < n_blocks; block_i += stages_count * gridDim.x) {
        auto block_begin = block_i * matrices_per_block;
        auto remaining_matrices = cuda::std::min(BlockSize, n_matrices - block_begin);

        auto pass_offset = block_begin * matrix_size;
        auto pass_size = remaining_matrices * matrix_size;
        auto *pass_a = &a[pass_offset];
        auto *pass_b = &b[pass_offset];
        auto *pass_c = &c[pass_offset];

        auto seq_a = seq % 2;
        auto seq_b = (seq + 1) % 2;
        // ++seq;

        pipeline.producer_acquire();
        cuda::memcpy_async(group, &shared_mem[seq_a][0], pass_a, pass_size * sizeof(Sca), pipeline);
        pipeline.producer_commit();

        pipeline.producer_acquire();
        cuda::memcpy_async(group, &shared_mem[seq_b][0], pass_b, pass_size * sizeof(Sca), pipeline);
        pipeline.producer_commit();

        pipeline.consumer_wait();
        auto* a_loc = &shared_mem[seq_a][threadIdx.x * matrix_size];
        for (int i=0; i<matrix_dim; ++i) {
            for (int j=0; j<matrix_dim; ++j) {
                A_tile(i, j) = a_loc[i * matrix_dim + j];
            }
        }
        pipeline.consumer_release();

        pipeline.consumer_wait();
        auto* b_loc = &shared_mem[seq_a][threadIdx.x * matrix_size];
        for (int i=0; i<matrix_dim; ++i) {
            for (int j=0; j<matrix_dim; ++j) {
                B_tile(i, j) = b_loc[i * matrix_dim + j];
            }
        }

        C_tile = A_tile * B_tile;

        for (int i=0; i<matrix_dim; ++i) {
            for (int j=0; j<matrix_dim; ++j) {
                b_loc[i * matrix_dim + j] = alpha * C_tile(i, j);
            }
        }


        auto* out = c + block_begin * matrix_size;
        for (unsigned i=threadIdx.x; i<remaining_matrices; i+=BlockSize) {
            out[i] = beta * out[i] + b_loc[i];
        }

        pipeline.consumer_release();


    }
}


#endif //TINY_BATCHED_GEMM_CUH
