//
//

#ifndef VECTORIZED_COPY_CUH
#define VECTORIZED_COPY_CUH

#include <cub/cub.cuh>

template <typename T, unsigned MatrixDim, unsigned BlockSize>
__device__ __forceinline__
void vectorized_copy(T* __restrict dst_ptr, const T* __restrict src_ptr, unsigned n_matrices) {

    constexpr auto vector_size = 16 / sizeof(T); // vectors are 128 bits
    constexpr auto matrix_size = MatrixDim * MatrixDim;
    constexpr auto block_size = BlockSize;
    constexpr auto smem_size = BlockSize * matrix_size;

    const auto n_vector_loads = n_matrices / vector_size;

    using Vector = cub::CubVector<T, vector_size>;

    auto* in_ptr = reinterpret_cast<const Vector *>(src_ptr);
    auto* out_ptr = reinterpret_cast<Vector *>(dst_ptr);

    for (unsigned i=threadIdx.x; i<n_vector_loads; i+=blockDim.x) {
        out_ptr[i] = in_ptr[i];
    }

    auto loaded = n_vector_loads * vector_size;
    for (unsigned i=loaded; i<n_matrices; i+=blockDim.x) {
        dst_ptr[i] = src_ptr[i];
    }
}

#endif //VECTORIZED_COPY_CUH
