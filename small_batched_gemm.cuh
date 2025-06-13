//
//

#ifndef SMALL_BATCHED_GEMM_CUH
#define SMALL_BATCHED_GEMM_CUH

#include <Eigen/Core>


#include "vectorized_copy.cuh"

/*
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


    auto load = [&](Matrix& dst, const Sca* src) {
        vectorized_copy<Sca, MatrixDim, BlockSize>(shared_mem, src, matrices_per_block);
        __syncthreads();
        for (int i=0; i<matrix_dim; ++i) {
            for (int j=0; j<matrix_dim; ++j) {
                dst(i, j) = shared_mem[
                    warp_id * full_matrix_size + full_matrix_dim * (block_row * matrix_dim + i) + block_col * matrix_dim + j
                    ];
            }
        }
        __syncthreads();
    };

    auto store = [&](const Matrix& src, Sca* dst) {
        for (int i=0; i<matrix_dim; ++i) {
            for (int j=0; j<matrix_dim; ++j) {
                shared_mem[
                    warp_id * full_matrix_size + full_matrix_dim * (block_row * matrix_dim + i) + block_col * matrix_dim + j
                    ] = src(i, j);
            }
        }
        __syncthreads();
        vectorized_copy<Sca, MatrixDim, BlockSize>(dst, shared_mem, matrices_per_block);
        __syncthreads();
    };


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
        load(A_tile, start_of_a_tile);
        load(B_tile, start_of_b_tile);

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
        load(A_tile, start_of_c_tile);
        B_tile.noalias() = beta*A_tile + alpha*C_tile;
        // write B_tile back to C
        store(B_tile, start_of_c_tile);
    }
}

#endif //SMALL_BATCHED_GEMM_CUH
