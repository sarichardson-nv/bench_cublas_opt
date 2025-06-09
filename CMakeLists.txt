cmake_minimum_required(VERSION 3.31)
project(bench_cublas CUDA CXX)

set(CMAKE_CUDA_STANDARD 20)


find_package(Threads REQUIRED)

find_package(benchmark CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(Thrust CONFIG REQUIRED)

find_package(NvidiaCutlass CONFIG)



add_executable(bench_cublas
        main.cu)

target_compile_options(bench_cublas PUBLIC
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)


set_target_properties(bench_cublas PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(bench_cublas PRIVATE
        benchmark::benchmark
        Eigen3::Eigen
        CUDA::cublas
        Thrust::Thrust
)

if (TARGET nvidia::cutlass::cutlass)
    # If CUTLASS is available, also build the cutlass benchmark

    target_link_libraries(bench_cublas PRIVATE nvidia::cutlass::cutlass)
    target_compile_definitions(bench_cublas PRIVATE
            USE_CUTLASS=1
    )

endif()


add_executable(understanding_loaders understand_cub_loaders.cu)


target_link_libraries(understanding_loaders PRIVATE Thrust::Thrust)