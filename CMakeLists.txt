cmake_minimum_required(VERSION 3.31)
project(bench_cublas CUDA CXX)

set(CMAKE_CUDA_STANDARD 20)
set(benchmark_DIR /home/scratch.sarichardson_gpu/oxford-math/benchmark/build)
set(Eigen3_DIR /home/scratch.sarichardson_gpu/oxford-math/eigen-3.4.0/eigen-3.4.0/cmake)
set(CMAKE_MODULE_PATH /home/scratch.sarichardson_gpu/oxford-math/eigen-3.4.0/eigen-3.4.0)

find_package(Threads REQUIRED)

find_package(benchmark CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)
include_directories(/home/scratch.sarichardson_gpu/oxford-math/eigen-3.4.0/eigen-3.4.0)


find_package(NvidiaCutlass CONFIG)



add_executable(bench_cublas
        main.cu
        bench_config.cuh
        tiny_batched_gemm.cuh
        tiny_batched_gemm_cls.cuh
        tiny_batched_gemm_no_coalesc.cuh
        small_batched_gemm.cuh
        vectorized_copy.cuh
        bench_tiny_batched_gemm_3x3.cu
        bench_tiny_batched_gemm_cls_3x3.cu
        bench_cublas_3x3.cu
        bench_tiny_batched_gemm_3x3_nocoal.cu
        bench_cublas_3x3.cu
        bench_tiny_batched_gemm_3x3_pipeline.cu
        bench_tiny_batched_gemm_4x4.cu
        bench_small_batched_gemm_4x4.cu
        bench_small_batched_gemm_pipelined_4x4.cu
        bench_cublas_4x4.cu
        bench_tiny_batched_gemm_6x6.cu
        bench_small_batched_gemm_6x6.cu
        bench_cublas_6x6.cu
        recursive_matrix.cuh
        bench_small_batched_gemm_12x12.cu
        bench_cublas_12x12.cu
        bench_small_batched_gemm_16x16.cu
        bench_cublas_16x16.cu
)

target_compile_options(bench_cublas PUBLIC
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --generate-line-info -arch=sm_86 --use_fast_math -maxrregcount 32>)




set_target_properties(bench_cublas PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(bench_cublas PRIVATE
        benchmark::benchmark
        CUDA::cublas
)

if (TARGET nvidia::cutlass::cutlass)
    # If CUTLASS is available, also build the cutlass benchmark

    target_link_libraries(bench_cublas PRIVATE nvidia::cutlass::cutlass)

    target_sources(bench_cublas PRIVATE bench_cutlass_3x3.cu)

endif()


add_executable(understanding_loaders understand_cub_loaders.cu)
