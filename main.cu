#include <benchmark/benchmark.h>

#include "recursive_matrix.cuh"


int main(int argc, char **argv) {
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    char arg0_default[] = "benchmark";
    char *args_default = reinterpret_cast<char *>(arg0_default);
    if (!argv) {
        argc = 1;
        argv = &args_default;
    }
    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();

    cudaDeviceReset();
    return 0;
}
