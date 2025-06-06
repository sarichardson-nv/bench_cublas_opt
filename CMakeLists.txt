cmake_minimum_required(VERSION 3.31)
project(bench_cublas CUDA CXX)

set(CMAKE_CUDA_STANDARD 20)


find_package(Threads REQUIRED)

find_package(benchmark CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(Thrust CONFIG REQUIRED)



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
