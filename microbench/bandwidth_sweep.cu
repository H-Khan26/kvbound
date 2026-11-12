// Memory bandwidth versus working-set size.
//
// Sweeps a buffer from kilobytes to gigabytes and measures how many bytes per second the GPU can
// read, write and copy. Small working sets live in cache and run far above the DRAM rate; once the
// working set outgrows L2 the rate drops to what the memory bus delivers. That plateau is the
// ceiling on decode speed: a decode step reads every weight once.
//
// Every thread loads 16 bytes at a time, and the 32 threads of a warp touch 512 consecutive
// bytes, so accesses are fully coalesced. Each thread owns a fixed set of elements for the whole
// run, so no two threads ever read the same bytes and one thread's cache fill cannot serve
// another. A consequence: each SM re-reads the same 1/sm_count share of the buffer every pass, so
// working sets up to sm_count times the L1 size are served by the L1s, and the drop to the DRAM
// rate appears at max(L2 size, sm_count * L1 size). pointer_chase isolates the L2 size itself.
//
// Each point is measured two ways. The sustained rate times launches of about 0.1 s; it is what
// a long-running workload gets, including time lost to whatever else uses the GPU (a desktop
// compositor, a browser). The burst rate is the best of many 2 ms launches, short enough that
// some finish without being interrupted; it is what the hardware delivers.

#include <cstdint>
#include <cstring>

#include "common.h"

namespace {

constexpr int kBlockThreads = 256;
constexpr int kBlocksPerSm = 6;  // 6 * 256 threads fills an SM's 1536 thread slots
constexpr double kTargetSeconds = 0.1;
constexpr int kRepeats = 5;
constexpr double kBurstSeconds = 0.002;
constexpr int kBursts = 40;

// Thread `tid` owns the elements congruent to tid modulo the thread count.
__device__ inline std::size_t thread_id() {
  return static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
}

__global__ void read_kernel(const uint4* buf, std::size_t n, int passes, unsigned long long* sink) {
  const std::size_t n_threads = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  unsigned acc = 0;
  for (int p = 0; p < passes; ++p) {
    for (std::size_t i = thread_id(); i < n; i += n_threads) {
      const uint4 v = buf[i];
      // Order-dependent mix, so no pass can be folded into another.
      acc = acc * 2654435761u + (v.x ^ v.y ^ v.z ^ v.w);
    }
  }
  atomicAdd(sink, static_cast<unsigned long long>(acc));
}

__global__ void write_kernel(uint4* buf, std::size_t n, int passes) {
  const std::size_t n_threads = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (int p = 0; p < passes; ++p) {
    for (std::size_t i = thread_id(); i < n; i += n_threads) {
      const unsigned v = (static_cast<unsigned>(i) + p) * 0x9E3779B1u;
      buf[i] = make_uint4(v, v ^ 0x85EBCA77u, v >> 7, v * 31u);
    }
  }
}

__global__ void copy_kernel(uint4* dst, const uint4* src, std::size_t n, int passes) {
  const std::size_t n_threads = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (int p = 0; p < passes; ++p) {
    for (std::size_t i = thread_id(); i < n; i += n_threads) {
      dst[i] = src[i];
    }
  }
}

// Fills the buffer with a different pseudo-random word per element (a 32-bit integer hash).
__global__ void fill_kernel(std::uint32_t* words, std::size_t n_words) {
  const std::size_t n_threads = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  const std::size_t tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  for (std::size_t i = tid; i < n_words; i += n_threads) {
    std::uint32_t h = static_cast<std::uint32_t>(i) * 0x9E3779B1u;
    h ^= h >> 15;
    h *= 0x85EBCA77u;
    h ^= h >> 13;
    words[i] = h;
  }
}

enum class Mode { Read, Write, Copy };
const char* mode_name(Mode m) {
  return m == Mode::Read ? "read" : m == Mode::Write ? "write" : "copy";
}

struct Point {
  std::size_t working_set_bytes;
  int threads;
  double best_gbs;    // sustained
  double median_gbs;  // sustained
  double burst_gbs;
};

// Measures one (mode, size). `working_set` is the total bytes touched per pass: one buffer for
// read and write, source plus destination for copy.
Point measure(Mode mode, std::size_t working_set, int max_blocks, uint4* a, uint4* b,
              unsigned long long* sink) {
  const std::size_t buffer_bytes = mode == Mode::Copy ? working_set / 2 : working_set;
  std::size_t n = buffer_bytes / sizeof(uint4);
  const int blocks = static_cast<int>(
      std::min<std::size_t>(max_blocks, std::max<std::size_t>(1, n / kBlockThreads)));
  const std::size_t n_threads = static_cast<std::size_t>(blocks) * kBlockThreads;
  n -= n % n_threads;  // every thread then makes the same number of loads per pass
  const double bytes_per_pass =
      static_cast<double>(n) * sizeof(uint4) * (mode == Mode::Copy ? 2.0 : 1.0);

  const auto launch = [&](int passes) {
    switch (mode) {
      case Mode::Read:
        read_kernel<<<blocks, kBlockThreads>>>(a, n, passes, sink);
        break;
      case Mode::Write:
        write_kernel<<<blocks, kBlockThreads>>>(a, n, passes);
        break;
      case Mode::Copy:
        copy_kernel<<<blocks, kBlockThreads>>>(b, a, n, passes);
        break;
    }
    CUDA_CHECK(cudaGetLastError());
  };

  // Calibrates the pass count so one timed launch runs for about kTargetSeconds, long enough
  // that launch overhead and clock ramp-up do not matter.
  int passes = 1;
  double seconds = kvb::bench::time_gpu([&] { launch(passes); });
  while (seconds < kTargetSeconds / 4 && passes < (1 << 28)) {
    passes *= 4;
    seconds = kvb::bench::time_gpu([&] { launch(passes); });
  }
  passes = std::max(1, static_cast<int>(passes * kTargetSeconds / seconds));

  std::vector<double> rates;
  for (int r = 0; r < kRepeats; ++r) {
    const double t = kvb::bench::time_gpu([&] { launch(passes); });
    rates.push_back(bytes_per_pass * passes / t / 1e9);
  }
  const int burst_passes = std::max(1, static_cast<int>(passes * kBurstSeconds / kTargetSeconds));
  double burst = 0.0;
  for (int r = 0; r < kBursts; ++r) {
    const double t = kvb::bench::time_gpu([&] { launch(burst_passes); });
    burst = std::max(burst, bytes_per_pass * burst_passes / t / 1e9);
  }
  return {static_cast<std::size_t>(bytes_per_pass), static_cast<int>(n_threads),
          *std::max_element(rates.begin(), rates.end()), kvb::bench::median(rates), burst};
}

}  // namespace

int main(int argc, char** argv) {
  // Largest working set in MiB; the default leaves room for a desktop on an 8 GB card.
  const std::size_t max_mib = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 2048;
  const std::size_t max_bytes = max_mib << 20;

  const cudaDeviceProp prop = kvb::bench::device_properties();
  const int max_blocks = prop.multiProcessorCount * kBlocksPerSm;

  uint4 *a = nullptr, *b = nullptr;
  unsigned long long* sink = nullptr;
  CUDA_CHECK(cudaMalloc(&a, max_bytes));
  CUDA_CHECK(cudaMalloc(&b, max_bytes / 2));
  CUDA_CHECK(cudaMalloc(&sink, sizeof(*sink)));
  fill_kernel<<<max_blocks, kBlockThreads>>>(reinterpret_cast<std::uint32_t*>(a), max_bytes / 4);
  fill_kernel<<<max_blocks, kBlockThreads>>>(reinterpret_cast<std::uint32_t*>(b), max_bytes / 8);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemset(sink, 0, sizeof(*sink)));

  std::printf("{\n  \"benchmark\": \"bandwidth_sweep\",\n");
  kvb::bench::print_device_json(prop);
  std::printf(
      "  \"params\": {\"element_bytes\": 16, \"block_threads\": %d, \"max_blocks\": %d, "
      "\"target_seconds\": %.2f, \"repeats\": %d, \"burst_seconds\": %.3f, \"bursts\": %d},\n"
      "  \"points\": [",
      kBlockThreads, max_blocks, kTargetSeconds, kRepeats, kBurstSeconds, kBursts);

  bool first = true;
  // Four sizes per octave from 16 KiB: 1, 1.19, 1.41, 1.68 times each power of two.
  for (double size = 16.0 * 1024; size <= static_cast<double>(max_bytes); size *= 1.189207115) {
    const auto working_set = static_cast<std::size_t>(size);
    for (const Mode mode : {Mode::Read, Mode::Write, Mode::Copy}) {
      const Point p = measure(mode, working_set, max_blocks, a, b, sink);
      std::printf(
          "%s\n    {\"mode\": \"%s\", \"working_set_bytes\": %zu, \"threads\": %d, "
          "\"best_gbs\": %.2f, \"median_gbs\": %.2f, \"burst_gbs\": %.2f}",
          first ? "" : ",", mode_name(mode), p.working_set_bytes, p.threads, p.best_gbs,
          p.median_gbs, p.burst_gbs);
      first = false;
    }
  }
  std::printf("\n  ]\n}\n");

  CUDA_CHECK(cudaFree(a));
  CUDA_CHECK(cudaFree(b));
  CUDA_CHECK(cudaFree(sink));
  return 0;
}
