//
//

#ifndef SMALL_BATCHED_GEMM_CUH
#define SMALL_BATCHED_GEMM_CUH

#include <Eigen/Core>

#include "vectorized_copy.cuh"
#include <cooperative_groups.h>
#include <random>
#include <cuda/pipeline>


namespace cg = cooperative_groups;


/*
 * @brief compute batched gemm operations on small matrices cooperatively
 *
 * This is Mike Giles' algorithm for computing batched products of matrices using
   cooperative thread groups. His original code was written in CUDA-C, so I've
 * translated it to C++ and templates and some other C++ niceties to simplify
 * the code.
 *
 * Unfortunately, there are some memory access issues with this at the moment that
 * I have yet to fix. Nonetheless I think it is important to include this here as
 * a demonstration of how larger matrices can be handled.
 */
template<typename Sca, int MatrixDim, int NThreadsPerDim, int BlockSize = 256, int Flags = Eigen::RowMajor>
__global__ void small_batched_cooperative_pipelined_gemm(Sca *__restrict__ a, Sca *__restrict__ b, Sca *__restrict__ c,
                                                         float alpha, float beta,
                                                         unsigned n_matrices) {
    static_assert(NThreadsPerDim > 0 && NThreadsPerDim < 32 && (NThreadsPerDim & (NThreadsPerDim - 1)) == 0,
                  "NThreadsPerDim must be a power of 2 and < 32"
    );
    constexpr int matrix_dim = MatrixDim;
    constexpr int matrix_size = MatrixDim * MatrixDim;

    constexpr int tile_dim = (MatrixDim + NThreadsPerDim - 1) / NThreadsPerDim;
    // constexpr int tile_size = tile_dim * tile_dim;

    constexpr int threads_per_dim = NThreadsPerDim;
    constexpr int threads_per_matrix = NThreadsPerDim * NThreadsPerDim;
    constexpr int matrices_per_block = BlockSize / threads_per_matrix;


    constexpr auto scope = cuda::thread_scope_block;
    constexpr auto smem_bank_size = matrices_per_block * matrix_size;
    constexpr auto num_smem_banks = 2;

    // ReSharper disable once CppTooWideScope
    constexpr unsigned thread_mask = 0xFF'FF'FF'FF;

    using Matrix = Eigen::Matrix<Sca, tile_dim, tile_dim, Flags>;

    auto this_block = cg::this_thread_block();

    __shared__ cuda::pipeline_shared_state<scope, 2> shared_state;
    __shared__ alignas(32) Sca shared_mem[num_smem_banks][smem_bank_size];


    // Which matrix is this thread working on
    const auto matrix_idx = threadIdx.x / threads_per_matrix;

    // which tile in the matrix am I working on, raw index
    const auto tile_idx = threadIdx.x % threads_per_matrix;

    // Where does that tile actually appear
    const auto tile_row = tile_idx / NThreadsPerDim;
    const auto tile_col = tile_idx % NThreadsPerDim;
    auto lane1 = tile_col + tile_col * threads_per_dim;
    auto lane2 = (tile_idx + 1) % threads_per_matrix + threads_per_matrix * tile_row;

    auto pipeline = cuda::make_pipeline(this_block, &shared_state);

    constexpr auto processed_per_block = matrices_per_block;

    const auto n_passes = (n_matrices + processed_per_block - 1) / processed_per_block;
    // constexpr auto pass_stride = matrix_size * matrices_per_block;


    Matrix A_tile, B_tile, C_tile;

    int seq = 0;
    for (unsigned pass_idx = blockIdx.x; pass_idx < n_passes; pass_idx += gridDim.x) {
        auto start_of_pass = pass_idx * processed_per_block;
        auto matrices_remaining = std::min(static_cast<unsigned>(matrices_per_block), n_matrices - start_of_pass);

        auto pass_offset = start_of_pass * matrix_size;
        const auto *start_of_a_block = a + pass_offset;
        const auto *start_of_b_block = b + pass_offset;
        auto *start_of_c_block = c + pass_offset;

        auto a_seq = seq % 2;
        auto b_seq = (seq + 1) % 2;
        // ++seq;

        pipeline.producer_acquire();
        cuda::memcpy_async(this_block,
            &shared_mem[a_seq][0],
            start_of_a_block,
            sizeof(Sca) * matrix_size * matrices_remaining,
            pipeline
            );
        pipeline.producer_commit();

        pipeline.producer_acquire();
        cuda::memcpy_async(this_block,
            &shared_mem[b_seq][0],
            start_of_b_block,
            sizeof(Sca) * matrix_size * matrices_remaining,
            pipeline
            );
        pipeline.producer_commit();

        pipeline.consumer_wait();
        auto *a_loc = &shared_mem[a_seq][matrix_idx * matrix_size];
        for (int i=0; i<tile_dim; ++i) {
            for (int j=0; j<tile_dim; ++j) {
                A_tile(i, j) = a_loc[matrix_dim*(tile_dim * tile_row + i) + (tile_dim*tile_col + j)];
            }
        }
        pipeline.consumer_release();

        // pipeline.producer_acquire();
        // cuda::memcpy_async(this_block,
        //     &shared_mem[a_seq][0],
        //     start_of_c_block,
        //     sizeof(Sca) * matrix_size * matrices_remaining,
        //     pipeline
        //     );
        // pipeline.producer_commit();


        pipeline.consumer_wait();
        auto *b_loc = &shared_mem[b_seq][matrix_idx * matrix_size];
        for (int i=0; i<tile_dim; ++i) {
            for (int j=0; j<tile_dim; ++j) {
                B_tile(i, j) = b_loc[matrix_dim*(tile_dim * tile_row + i) + (tile_dim*tile_col + j)];
            }
        }
        pipeline.consumer_release();


        C_tile.setZero();

        for (unsigned rotate = 0; rotate < threads_per_dim; rotate++) {
            for (unsigned j = 0; j < tile_dim; ++j) {
                for (unsigned k = 0; k < tile_dim; ++k) {
                    auto rhs_val = __shfl_sync(thread_mask, B_tile(k, j), lane1, threads_per_matrix);
                    for (unsigned i = 0; i < tile_dim; ++i) {
                        C_tile(i, j) += A_tile(i, k) * rhs_val;
                    }
                }
            }
            lane1 = (lane1 + threads_per_dim) % threads_per_matrix;
            for (unsigned j = 0; j < tile_dim; ++j) {
                for (unsigned i = 0; i < tile_dim; ++i) {
                    A_tile(i, j) = __shfl_sync(thread_mask, A_tile(i, j), lane2, threads_per_matrix);
                }
            }
        } // end of block loop


        // pipeline.consumer_wait();
        // auto *c_loc = &shared_mem[a_seq][matrix_idx * matrix_size];
        auto *out = start_of_c_block + matrix_idx * matrix_size;
        for (int i=0; i<tile_dim; ++i) {
            for (int j=0; j<tile_dim; ++j) {
                auto& out_val = out[matrix_dim*(tile_dim * tile_row + i) + (tile_dim*tile_col + j)];
                auto c_val = alpha*out_val;
                // auto c_val = alpha * c_loc[matrix_dim*(tile_dim * tile_row + i) + (tile_dim*tile_col + j)];
                // out[matrix_dim * (tile_dim * tile_row + i) + tile_col + j] = alpha * c_val + beta * C_tile(i, j);
                out_val = c_val + beta * C_tile(i, j);
            }
        }
        //
        // this_block.sync();
        //
        // for (unsigned i=threadIdx.x; i<matrices_remaining*matrix_size; i+=blockDim.x) {
        //     start_of_c_block[i] = shared_mem[a_seq][i];
        // }

        // pipeline.consumer_release();

    }
}

#endif //SMALL_BATCHED_GEMM_CUH
