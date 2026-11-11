#pragma once

// Shared by the standalone microbenchmarks: CUDA error checks, wall-clock timing of GPU work, and
// the device block of the JSON each benchmark prints to stdout. The Python wrapper
// (kvbound.microbench) adds the machine fingerprint and writes the result record.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                                    \
  do {                                                                                      \
    const cudaError_t kvb_err = (call);                                                     \
    if (kvb_err != cudaSuccess) {                                                           \
      std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(kvb_err)); \
      std::exit(1);                                                                         \
    }                                                                                       \
  } while (0)

namespace kvb::bench {

// Seconds taken by `work` including all GPU work it launches.
template <typename F>
double time_gpu(F&& work) {
  CUDA_CHECK(cudaDeviceSynchronize());
  const auto start = std::chrono::steady_clock::now();
  work();
  CUDA_CHECK(cudaDeviceSynchronize());
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

inline double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const std::size_t mid = values.size() / 2;
  return values.size() % 2 ? values[mid] : 0.5 * (values[mid - 1] + values[mid]);
}

inline cudaDeviceProp device_properties() {
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  return prop;
}

// Prints `"device": {...},` as the first member of the benchmark's JSON object.
inline void print_device_json(const cudaDeviceProp& prop) {
  std::printf(
      "  \"device\": {\"name\": \"%s\", \"compute_capability\": \"%d.%d\", "
      "\"sm_count\": %d, \"l2_cache_bytes\": %d, \"total_memory_bytes\": %zu},\n",
      prop.name, prop.major, prop.minor, prop.multiProcessorCount, prop.l2CacheSize,
      prop.totalGlobalMem);
}

}  // namespace kvb::bench
