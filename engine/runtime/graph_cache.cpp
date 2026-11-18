#include "kvbound/cuda/graph_cache.h"

#include <algorithm>
#include <chrono>
#include <string>

#include "kvbound/cuda/device.h"

namespace kvb::cuda {

void GraphCache::run(std::int64_t key, cudaStream_t stream, const std::function<void()>& launch) {
  auto it = graphs_.find(key);
  if (it == graphs_.end()) {
    const auto start = std::chrono::steady_clock::now();
    if (static_cast<int>(graphs_.size()) >= max_graphs_) {
      const auto oldest = std::min_element(
          graphs_.begin(), graphs_.end(),
          [](const auto& a, const auto& b) { return a.second.last_used < b.second.last_used; });
      KVB_CUDA(cudaGraphExecDestroy(oldest->second.exec));
      graphs_.erase(oldest);
      stats_.evicted += 1;
    }
    cudaGraph_t graph = nullptr;
    KVB_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    try {
      launch();
    } catch (...) {
      cudaStreamEndCapture(stream, &graph);
      if (graph != nullptr) cudaGraphDestroy(graph);
      throw;
    }
    KVB_CUDA(cudaStreamEndCapture(stream, &graph));
    cudaGraphExec_t exec = nullptr;
    const cudaError_t status = cudaGraphInstantiate(&exec, graph, 0);
    cudaGraphDestroy(graph);
    KVB_CUDA(status);
    it = graphs_.emplace(key, Graph{.exec = exec}).first;
    stats_.captured += 1;
    stats_.capture_s +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  }
  KVB_CUDA(cudaGraphLaunch(it->second.exec, stream));
  stats_.replayed += 1;
  it->second.last_used = stats_.replayed;
}

void GraphCache::step(bool graphs, int n_tokens, int n_out, int n_tiles, int max_rows, bool hooked,
                      cudaStream_t stream, const std::function<void(bool graphed)>& launch,
                      int variant) {
  if (variant < 0 || variant >= kMaxGraphVariants) {
    throw Error("a graph variant is between 0 and " + std::to_string(kMaxGraphVariants - 1));
  }
  // Kernel arguments are frozen in a graph, so it serves only steps whose launches depend on
  // nothing but the row count: never the tiled attention, never a hook's synchronization.
  if (graphs && !hooked && n_tiles == 0 && n_out == n_tokens && n_tokens <= max_rows &&
      n_tokens < kMaxGraphRows) {
    const std::int64_t key = static_cast<std::int64_t>(variant) << 40 |
                             static_cast<std::int64_t>(n_tokens) << 20 | n_out;
    run(key, stream, [&] { launch(true); });
    return;
  }
  stats_.eager += 1;
  launch(false);
}

void GraphCache::clear() noexcept {
  for (const auto& [key, graph] : graphs_) cudaGraphExecDestroy(graph.exec);
  graphs_.clear();
}

}  // namespace kvb::cuda
