# Batched matrix operations for small matrices

This repository contains a number of benchmarks of various methods of computing a large number of matrix products in
parallel. The standard here is the cuBLAS strided batch GEMM operation. This also includes a custom kernel that uses
template matrix dimensions and performs the computations using a one matrix per thread model. In this case, loading and
storing the matrices to the GPU global memory is the most problematic aspect.

The motivation for these benchmarks come from recent work on Linear Neural Controlled Differential Equations (LNCDEs);
see for instance <https://arxiv.org/abs/2505.17761>. Performing a single step evaluation of such a model involves
computing the matrix exponential of a large number of tiny matrices (for instance 2x2 or 3x3), which presents an
interesting challenge. Using a (simplified) Pade approximation scheme based on the classic scale and squaring method
requires a large number of registers to hold the intermediate values, which limits the GPU occupancy. Coupling this
restriction with the need for very high throughput means that we need very carefully designed kernels that maximize
compute performance. This lead to experimentation with templated kernels parametrised by the matrix dimension (and
scalar type), and eventually to these benchmarks.

## The benchmarks

The benchmarks presented here compare the performance of a batched GEMM operation for 3x3 matrices in both row-major and
column-major format with float32 coefficients. The suffix `_rm` denotes row-major and `_cm` denotes column major. There
are four different strategies for performing this calculation. The first is using cuBLAS `cublasSgemmStridedBatched`
routine. The second uses our `tiny_batched_gemm` kernel, which uses a one-matrix-per-thread and templated matrix size
to perform the computation. We include two versions of this kernel, one which uses the `cub::BlockLoad` and
`cub::BlockStore` primitives and one that uses a custom vectorised load and store operaton (denoted by the `_cls`
suffix). The next two use the CUTLASS library and the `GemmBatched` construction. The final benchmark is not currently
included in any of the results, is the `small_batched_gemm` which uses a sub-warp block of threads to compute each
matrix calculation cooperatively.

In all cases, we present results of the benchmark on a batch of size 8192. We tested this on two different platforms: a
consumer RTX 3070Ti and a A100. The following results are generated on a RTX 3070Ti (CUDA toolkit 12.9.86).

```text
----------------------------------------------------------------------------------
Benchmark                                        Time             CPU   Iterations
----------------------------------------------------------------------------------
bench_cublas_3x3_rm/8192                    455909 ns       454029 ns        10000
bench_cublas_3x3_cm/8192                     29025 ns        28892 ns        24844
bench_tiny_batched_gemm_3x3_rm/8192           6250 ns         6221 ns       112498
bench_tiny_batched_gemm_3x3_cm/8192           6294 ns         6258 ns       113590
bench_tiny_batched_gemm_3x3_rm_cls/8192       5692 ns         5665 ns       123229
bench_cutlass_3x3_rm/8192                   413159 ns       411386 ns        10000
bench_cutlass_3x3_rm/8192                   420357 ns       418251 ns        10000
```

The results on the A100 are below (CUDA toolkit 12.6.85). CUTLASS benchmarks are omitted for now.

```text
----------------------------------------------------------------------------------
Benchmark                                        Time             CPU   Iterations
----------------------------------------------------------------------------------
bench_cublas_3x3_rm/8192                    285332 ns       285326 ns        11016
bench_cublas_3x3_cm/8192                    117959 ns       117951 ns        10000
bench_tiny_batched_gemm_3x3_rm/8192          11013 ns        11013 ns        63856
bench_tiny_batched_gemm_3x3_cm/8192          11013 ns        11012 ns        63838
bench_tiny_batched_gemm_3x3_rm_cls/8192       9490 ns         9489 ns        73529
```

