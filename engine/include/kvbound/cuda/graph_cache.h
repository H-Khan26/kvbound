#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <functional>
#include <unordered_map>

namespace kvb::cuda {

// How an engine launched its steps so far.
struct GraphStats {
  std::int64_t replayed = 0;  // steps launched from a graph
  std::int64_t eager = 0;     // steps launched kernel by kernel
  int captured = 0;           // graphs built, counting a step shape again after its eviction
  int evicted = 0;
  double capture_s = 0.0;  // time spent capturing and instantiating
};

// CUDA graphs of an engine's steps, one per step shape. A captured graph replays every kernel
// launch of a step in one call. Kernel arguments are frozen at capture time, which is safe
// because everything that changes between steps (tokens, positions, cache slots, block tables)
// lives in device buffers at fixed addresses. At most max_graphs graphs are kept; the least
// recently used one is dropped when a new shape needs one.
class GraphCache {
 public:
  explicit GraphCache(int max_graphs) : max_graphs_(max_graphs) {}
  ~GraphCache() { clear(); }
  GraphCache(const GraphCache&) = delete;
  GraphCache& operator=(const GraphCache&) = delete;

  // Launches one step of n_tokens rows, n_out of them outputs. A decode-shaped step (every row
  // an output, at most max_rows rows, no attention tiles) replays the graph of its shape when
  // `graphs` is set and nothing is hooked; any other step runs eagerly. launch(graphed) issues
  // the step's kernels; a graphed launch takes no tiles and no hooks.
  void step(bool graphs, int n_tokens, int n_out, int n_tiles, int max_rows, bool hooked,
            cudaStream_t stream, const std::function<void(bool graphed)>& launch);
  const GraphStats& stats() const noexcept { return stats_; }
  // Destroys every graph.
  void clear() noexcept;

 private:
  // Launches the graph of step shape `key` on `stream`. A shape with no graph gets one first:
  // `launch` is called once while the stream captures, and must issue the step's kernels on
  // that stream and nothing else.
  void run(std::int64_t key, cudaStream_t stream, const std::function<void()>& launch);

  struct Graph {
    cudaGraphExec_t exec = nullptr;
    std::int64_t last_used = 0;  // value of stats_.replayed at the latest launch
  };

  int max_graphs_;
  std::unordered_map<std::int64_t, Graph> graphs_;
  GraphStats stats_;
};

}  // namespace kvb::cuda
