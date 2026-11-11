// Memory latency versus working-set size.
//
// One GPU thread follows a random cycle of pointers: each load's address comes from the previous
// load, so nothing can overlap and time per hop is the latency of one dependent memory access.
// Sweeping the working set shows the L1, L2 and DRAM latency plateaus. Bandwidth (the other
// microbenchmark) says how fast independent reads stream; this says what one unpredictable read
// costs, which is what a gather through a block table pays when it misses cache.

#include <cstdint>
#include <numeric>
#include <random>

#include "common.h"

namespace {

// Each node sits alone in a 128-byte cache line, so every hop touches a new line.
constexpr std::size_t kNodeBytes = 128;
constexpr std::size_t kNodeWords = kNodeBytes / sizeof(std::uint32_t);
constexpr std::uint32_t kMeasuredHops = 1u << 20;
constexpr int kRepeats = 3;

__global__ void chase_kernel(const std::uint32_t* nodes, std::uint32_t start, std::uint32_t hops,
                             std::uint32_t* end) {
  std::uint32_t node = start;
  for (std::uint32_t h = 0; h < hops; ++h)
    node = nodes[static_cast<std::size_t>(node) * kNodeWords];
  *end = node;
}

// Sattolo's algorithm: a uniformly random permutation that is one single cycle through all nodes.
std::vector<std::uint32_t> random_cycle(std::uint32_t n_nodes, std::mt19937_64& rng) {
  std::vector<std::uint32_t> next(n_nodes);
  std::iota(next.begin(), next.end(), 0u);
  for (std::uint32_t i = n_nodes - 1; i > 0; --i) {
    const auto j = static_cast<std::uint32_t>(rng() % i);
    std::swap(next[i], next[j]);
  }
  return next;
}

}  // namespace

int main(int argc, char** argv) {
  const std::size_t max_mib = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 1024;
  const std::size_t max_bytes = max_mib << 20;

  const cudaDeviceProp prop = kvb::bench::device_properties();
  std::uint32_t *nodes = nullptr, *end = nullptr;
  CUDA_CHECK(cudaMalloc(&nodes, max_bytes));
  CUDA_CHECK(cudaMalloc(&end, sizeof(*end)));

  std::printf("{\n  \"benchmark\": \"pointer_chase\",\n");
  kvb::bench::print_device_json(prop);
  std::printf(
      "  \"params\": {\"node_bytes\": %zu, \"measured_hops\": %u, \"repeats\": %d},\n"
      "  \"points\": [",
      kNodeBytes, kMeasuredHops, kRepeats);

  std::mt19937_64 rng(0);
  std::vector<std::uint32_t> host(max_bytes / sizeof(std::uint32_t), 0u);
  bool first = true;
  for (std::size_t bytes = 4096; bytes <= max_bytes; bytes *= 2) {
    const auto n_nodes = static_cast<std::uint32_t>(bytes / kNodeBytes);
    const std::vector<std::uint32_t> next = random_cycle(n_nodes, rng);
    for (std::uint32_t i = 0; i < n_nodes; ++i) host[i * kNodeWords] = next[i];
    CUDA_CHECK(cudaMemcpy(nodes, host.data(), bytes, cudaMemcpyHostToDevice));

    // One lap (capped) loads the working set into whatever cache level can hold it.
    const std::uint32_t warm_hops = std::min(n_nodes, kMeasuredHops);
    chase_kernel<<<1, 1>>>(nodes, 0, warm_hops, end);
    CUDA_CHECK(cudaGetLastError());

    std::vector<double> ns_per_hop;
    for (int r = 0; r < kRepeats; ++r) {
      const double t = kvb::bench::time_gpu([&] {
        chase_kernel<<<1, 1>>>(nodes, 0, kMeasuredHops, end);
        CUDA_CHECK(cudaGetLastError());
      });
      ns_per_hop.push_back(t * 1e9 / kMeasuredHops);
    }
    std::printf(
        "%s\n    {\"working_set_bytes\": %zu, \"best_ns_per_hop\": %.2f, "
        "\"median_ns_per_hop\": %.2f}",
        first ? "" : ",", bytes, *std::min_element(ns_per_hop.begin(), ns_per_hop.end()),
        kvb::bench::median(ns_per_hop));
    first = false;
  }
  std::printf("\n  ]\n}\n");

  CUDA_CHECK(cudaFree(nodes));
  CUDA_CHECK(cudaFree(end));
  return 0;
}
