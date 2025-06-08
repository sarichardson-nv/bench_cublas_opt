//
//



#include <cub/block/block_load.cuh>
#include <cub/warp/warp_load.cuh>

#include <thrust/universal_vector.h>
#include <thrust/sequence.h>



template <typename Loader>
__global__ void test_loader(unsigned const* __restrict__ d_input, unsigned* __restrict__ d_output) {

    __shared__ typename Loader::TempStorage temp_storage;

    unsigned thread_data[2];
    auto block_offset = 2*blockIdx.x * blockDim.x;

    Loader(temp_storage).Load(d_input + block_offset, thread_data);

    thread_data[0] = thread_data[0] % (2*blockDim.x*gridDim.x);
    thread_data[1] = thread_data[1] % (2*blockDim.x*gridDim.x);


    d_output[block_offset + 2*threadIdx.x] = (blockIdx.x << 16) | threadIdx.x;
    d_output[block_offset + 2*threadIdx.x+1] = (thread_data[0] << 16) | thread_data[1];

}

template <cub::WarpLoadAlgorithm Algorithm, int LogicalWarpThreads, int ItemsPerThread>
__global__ void test_loader_warp(unsigned const* __restrict__ d_input, unsigned* __restrict__ d_output) {

    constexpr int warps_in_block = 64 / LogicalWarpThreads;
    using Loader = cub::WarpLoad<unsigned, 2, Algorithm, LogicalWarpThreads>;

    __shared__ typename Loader::TempStorage temp_storage[warps_in_block];

    unsigned thread_data[ItemsPerThread];

    const auto lane = threadIdx.x % LogicalWarpThreads;
    const auto warp = threadIdx.x / LogicalWarpThreads;

    const auto block_offset = ItemsPerThread*blockIdx.x * blockDim.x;
    Loader(temp_storage[warp]).Load(d_input + block_offset + ItemsPerThread*LogicalWarpThreads*warp, thread_data);
    __syncthreads();

    thread_data[0] = thread_data[0] % (ItemsPerThread*blockDim.x*gridDim.x);
    thread_data[1] = thread_data[1] % (ItemsPerThread*blockDim.x*gridDim.x);

    d_output[block_offset + ItemsPerThread*threadIdx.x] = (blockIdx.x << 16) | threadIdx.x;
    d_output[block_offset + ItemsPerThread*threadIdx.x+1] = (thread_data[0] << 16) | thread_data[1];
}




int main() {
    const auto N = 32768;

    thrust::universal_vector<unsigned> input(N);
    thrust::universal_vector<unsigned> output(N);

    thrust::sequence(input.begin(), input.end(), 0);


    auto* in_ptr = raw_pointer_cast(input.data());
    auto* out_ptr = raw_pointer_cast(output.data());
    int no_algs = 0;


    unsigned threads = 64; // two warps in size
    unsigned blocks = N / 64; // One block

#if 0
    ++no_algs;
    test_loader<cub::BlockLoad<unsigned, 64, 2, cub::BLOCK_LOAD_DIRECT>><<<blocks, threads>>>(in_ptr, out_ptr);

    ++no_algs;
    in_ptr += 2*blocks*threads;
    out_ptr += 2*blocks*threads;
    test_loader<cub::BlockLoad<unsigned, 64, 2, cub::BLOCK_LOAD_VECTORIZE>><<<blocks, threads>>>(in_ptr, out_ptr);

    ++no_algs;
    in_ptr += 2*blocks*threads;
    out_ptr += 2*blocks*threads;
    test_loader<cub::BlockLoad<unsigned, 64, 2, cub::BLOCK_LOAD_TRANSPOSE>><<<blocks, threads>>>(in_ptr, out_ptr);

    ++no_algs;
    in_ptr += 2*blocks*threads;
    out_ptr += 2*blocks*threads;
    test_loader<cub::BlockLoad<unsigned, 64, 2, cub::BLOCK_LOAD_STRIPED>><<<blocks, threads>>>(in_ptr, out_ptr);

    ++no_algs;
    in_ptr += 2*blocks*threads;
    out_ptr += 2*blocks*threads;
    test_loader<cub::BlockLoad<unsigned, 64, 2, cub::BLOCK_LOAD_WARP_TRANSPOSE>><<<blocks, threads>>>(in_ptr, out_ptr);


    for (int i=0; i<no_algs; ++i) {
        printf("Algorithm %d\n", i);
        for (int block=0; block<blocks; ++block) {
            printf("b%d\n", block);
            for (int j=0; j<threads; ++j) {
                auto& first = output[(blocks*i + block)*2*threads+2*j];
                auto& second = output[(blocks*i + block)*2*threads+2*j+1];

                auto block_id = first >> 16;
                auto thread_id = first & 0xffff;
                auto val0 = second >> 16;
                auto val1 = second & 0xffff;

                printf("%2d %2d %5d %5d\n", thread_id, block_id, val0, val1);
            }
            printf("\n");
        }
    }

#endif

    ++no_algs;
    test_loader_warp<cub::WARP_LOAD_DIRECT, 2, 2><<<blocks, threads>>>(in_ptr, out_ptr);
    cudaDeviceSynchronize();

    // for (int i=0; i<no_algs; ++i) {
    //     printf("Algorithm %d\n", i);
    //     for (int block=0; block<blocks; ++block) {
    //         printf("b%d\n", block);
    //         for (int j=0; j<threads; ++j) {
    //             auto& first = output[(blocks*i + block)*2*threads+2*j];
    //             auto& second = output[(blocks*i + block)*2*threads+2*j+1];
    //
    //             auto block_id = first >> 16;
    //             auto thread_id = first & 0xffff;
    //             auto val0 = second >> 16;
    //             auto val1 = second & 0xffff;
    //
    //             printf("%2d %2d %5d %5d\n", thread_id, block_id, val0, val1);
    //         }
    //     }
    // }

    return 0;
}