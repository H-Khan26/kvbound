#include "kvbound/cuda/engine.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <numeric>

#include "kvbound/cuda/kernels.h"
#include "kvbound/dtype.h"

namespace kvb::cuda {
namespace {

EngineConfig resolved(EngineConfig config) {
  if (config.max_ctx <= 0 || config.max_seqs <= 0) throw Error("max_ctx and max_seqs must be > 0");
  if (config.max_step_tokens <= 0) config.max_step_tokens = config.max_ctx;
  if (config.max_outputs <= 0) config.max_outputs = config.max_seqs;
  // One step must be able to decode a token for every cached sequence.
  if (config.max_step_tokens < config.max_seqs || config.max_outputs < config.max_seqs) {
    throw Error("max_step_tokens and max_outputs must be at least max_seqs");
  }
  config.max_outputs = std::min(config.max_outputs, config.max_step_tokens);
  if (config.kv_block_size <= 0) config.kv_block_size = config.max_ctx;
  if (config.kv_block_size > config.max_ctx) throw Error("kv_block_size exceeds max_ctx");
  const int blocks_per_seq = (config.max_ctx + config.kv_block_size - 1) / config.kv_block_size;
  if (config.kv_blocks <= 0) config.kv_blocks = config.max_seqs * blocks_per_seq;
  if (config.kv_blocks < blocks_per_seq) throw Error("the KV pool cannot hold one full sequence");
  if (config.use_graph && config.max_graphs <= 0) throw Error("max_graphs must be > 0");
  return config;
}

cudaStream_t make_stream() {
  cudaStream_t stream = nullptr;
  KVB_CUDA(cudaStreamCreate(&stream));
  return stream;
}

}  // namespace

Engine::Engine(const std::filesystem::path& model_dir, const EngineConfig& config)
    : config_(resolved(config)),
      stream_(make_stream()),
      blocks_(config_.kv_blocks),
      rows_(config_.max_seqs),
      rows_host_(config_.max_step_tokens * sizeof(StepRow)),
      out_rows_host_(config_.max_step_tokens * sizeof(std::int32_t)),
      tiles_host_(config_.max_step_tokens * sizeof(AttnTile)),
      next_ids_(config_.max_step_tokens * sizeof(std::int32_t)),
      rows_dev_(rows_host_.bytes()),
      out_rows_dev_(out_rows_host_.bytes()),
      tiles_dev_(tiles_host_.bytes()) {
  model_ = std::make_unique<LlamaModel>(
      model_dir,
      LlamaModelConfig{.max_ctx = config_.max_ctx,
                       .max_step_tokens = config_.max_step_tokens,
                       .max_outputs = config_.max_outputs,
                       .plan = config_.plan,
                       .gemm = config_.gemm,
                       .fuse = config_.fuse,
                       .f16_gemv_rows = config_.f16_gemv_rows,
                       .wide_key_loads = config_.wide_key_loads,
                       .quantized_gemm_rows = config_.quantized_gemm_rows,
                       .expand_quantized = config_.expand_quantized,
                       .dequant_slice_bytes = config_.dequant_slice_bytes},
      stream_);
  const ModelConfig& cfg = model_->config();
  const int blocks_per_seq = (config_.max_ctx + config_.kv_block_size - 1) / config_.kv_block_size;
  kv_ = std::make_unique<KvPool>(KvDims{.n_layers = cfg.n_layers,
                                        .n_blocks = config_.kv_blocks,
                                        .n_kv_heads = cfg.n_kv_heads,
                                        .block_size = config_.kv_block_size,
                                        .head_dim = cfg.head_dim()},
                                 config_.kv_layout, config_.max_seqs, blocks_per_seq,
                                 config_.kv_dtype);
  KVB_CUDA(cudaStreamSynchronize(stream_));
}

Engine::~Engine() {
  for (const auto& [key, graph] : graphs_) cudaGraphExecDestroy(graph.exec);
  model_.reset();
  kv_.reset();
  if (stream_ != nullptr) cudaStreamDestroy(stream_);
}

EngineMemory Engine::memory() const {
  const std::vector<std::size_t>& sizes = model_->value_sizes();
  return EngineMemory{
      .weights = model_->weight_bytes(),
      .kv_pool = kv_->bytes(),
      .arena = model_->memory_plan().arena_bytes,
      .arena_naive = std::accumulate(sizes.begin(), sizes.end(), std::size_t{0}),
      .arena_peak_live = peak_live_bytes(sizes, value_lifetimes(model_->program())),
      .dequant = model_->dequant_bytes(),
  };
}

void Engine::set_blocks(int seq, std::span<const int> blocks) {
  if (seq < 0 || seq >= config_.max_seqs) throw Error("block-table row out of range");
  kv_->set_blocks(seq, blocks, stream_);
}

void Engine::forward(std::span<const StepRow> rows, std::span<const std::int32_t> out_rows,
                     const ForwardOptions& options) {
  const int n_tokens = static_cast<int>(rows.size());
  const int n_out = static_cast<int>(out_rows.size());
  if (n_tokens == 0 || n_tokens > config_.max_step_tokens || n_out > n_tokens ||
      n_out > config_.max_outputs) {
    throw Error("step has " + std::to_string(n_tokens) + " rows and " + std::to_string(n_out) +
                " outputs; the limits are " + std::to_string(config_.max_step_tokens) + " and " +
                std::to_string(config_.max_outputs));
  }
  for (const StepRow& row : rows) {
    if (row.position < 0 || row.position >= config_.max_ctx || row.seq < 0 ||
        row.seq >= config_.max_seqs || row.token_id < 0 ||
        row.token_id >= model_->config().vocab_size) {
      throw Error("step row is out of range");
    }
  }

  std::memcpy(rows_host_.as(), rows.data(), rows.size_bytes());
  std::memcpy(out_rows_host_.as(), out_rows.data(), out_rows.size_bytes());
  KVB_CUDA(cudaMemcpyAsync(rows_dev_.as(), rows_host_.as(), rows.size_bytes(),
                           cudaMemcpyHostToDevice, stream_));
  if (n_out > 0) {
    KVB_CUDA(cudaMemcpyAsync(out_rows_dev_.as(), out_rows_host_.as(), out_rows.size_bytes(),
                             cudaMemcpyHostToDevice, stream_));
  }

  // A step with a run of several rows of one sequence runs attention as tiles of those runs.
  int n_tiles = 0;
  if (config_.prefill_attention) {
    const std::vector<AttnTile> tiles = plan_attention_tiles(rows, kernels::kAttnTileRows);
    if (tiles.size() < rows.size()) {
      n_tiles = static_cast<int>(tiles.size());
      std::memcpy(tiles_host_.as(), tiles.data(), tiles.size() * sizeof(AttnTile));
      KVB_CUDA(cudaMemcpyAsync(tiles_dev_.as(), tiles_host_.as(), tiles.size() * sizeof(AttnTile),
                               cudaMemcpyHostToDevice, stream_));
    }
  }

  const bool hooked = options.capture != nullptr || options.times != nullptr;
  // Graphs are kept for decode-shaped steps only: few rows, every row an output, one row per
  // sequence. Their kernel arguments are frozen, so they never take the tiled attention.
  if (config_.use_graph && !hooked && n_tiles == 0 && n_out == n_tokens &&
      n_tokens <= config_.max_seqs) {
    run_graph(n_tokens, n_out);
  } else if (!hooked) {
    graph_stats_.eager += 1;
    model_->run(rows_dev_.as<StepRow>(), out_rows_dev_.as<std::int32_t>(), n_tokens, n_out,
                tiles_dev_.as<AttnTile>(), n_tiles, *kv_, stream_);
  } else {
    graph_stats_.eager += 1;
    RunHooks hooks;
    hooks.times = options.times;
    if (options.capture != nullptr) {
      hooks.on_site = [&](const std::string& site, int id, const void* data) {
        const Value& value = model_->program().values[id];
        const std::size_t n =
            static_cast<std::size_t>(value.rows == RowKind::Token ? n_tokens : n_out) * value.cols;
        std::vector<float>& host = (*options.capture)[site];
        host.resize(n);
        if (value.dtype == DType::F32) {
          KVB_CUDA(cudaMemcpyAsync(host.data(), data, n * sizeof(float), cudaMemcpyDeviceToHost,
                                   stream_));
          KVB_CUDA(cudaStreamSynchronize(stream_));
          return;
        }
        std::vector<std::uint16_t> bits(n);
        KVB_CUDA(cudaMemcpyAsync(bits.data(), data, n * sizeof(std::uint16_t),
                                 cudaMemcpyDeviceToHost, stream_));
        KVB_CUDA(cudaStreamSynchronize(stream_));
        std::transform(bits.begin(), bits.end(), host.begin(), half_bits_to_float);
      };
    }
    model_->run(rows_dev_.as<StepRow>(), out_rows_dev_.as<std::int32_t>(), n_tokens, n_out,
                tiles_dev_.as<AttnTile>(), n_tiles, *kv_, stream_, &hooks);
  }

  if (n_out > 0) {
    KVB_CUDA(cudaMemcpyAsync(next_ids_.as(), model_->next_ids(), n_out * sizeof(std::int32_t),
                             cudaMemcpyDeviceToHost, stream_));
  }
  KVB_CUDA(cudaStreamSynchronize(stream_));
  last_n_out_ = static_cast<std::size_t>(n_out);
}

// A captured graph replays every kernel launch of a step in one call. Kernel arguments are
// frozen at capture time, which is safe because everything that changes between steps (tokens,
// positions, cache slots, block tables) lives in device buffers at fixed addresses.
void Engine::run_graph(int n_tokens, int n_out) {
  const std::int64_t key = static_cast<std::int64_t>(n_tokens) << 32 | n_out;
  auto it = graphs_.find(key);
  if (it == graphs_.end()) {
    const auto start = std::chrono::steady_clock::now();
    if (static_cast<int>(graphs_.size()) >= config_.max_graphs) {
      const auto oldest = std::min_element(
          graphs_.begin(), graphs_.end(),
          [](const auto& a, const auto& b) { return a.second.last_used < b.second.last_used; });
      KVB_CUDA(cudaGraphExecDestroy(oldest->second.exec));
      graphs_.erase(oldest);
      graph_stats_.evicted += 1;
    }
    cudaGraph_t graph = nullptr;
    KVB_CUDA(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal));
    try {
      model_->run(rows_dev_.as<StepRow>(), out_rows_dev_.as<std::int32_t>(), n_tokens, n_out,
                  nullptr, 0, *kv_, stream_);
    } catch (...) {
      cudaStreamEndCapture(stream_, &graph);
      if (graph != nullptr) cudaGraphDestroy(graph);
      throw;
    }
    KVB_CUDA(cudaStreamEndCapture(stream_, &graph));
    cudaGraphExec_t exec = nullptr;
    const cudaError_t status = cudaGraphInstantiate(&exec, graph, 0);
    cudaGraphDestroy(graph);
    KVB_CUDA(status);
    it = graphs_.emplace(key, Graph{.exec = exec}).first;
    graph_stats_.captured += 1;
    graph_stats_.capture_s +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  }
  KVB_CUDA(cudaGraphLaunch(it->second.exec, stream_));
  graph_stats_.replayed += 1;
  it->second.last_used = graph_stats_.replayed;
}

std::vector<float> Engine::logits() {
  std::vector<float> host(last_n_out_ * static_cast<std::size_t>(model_->config().vocab_size));
  if (!host.empty()) {
    KVB_CUDA(cudaMemcpy(host.data(), model_->logits(), host.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  }
  return host;
}

Engine::Sequence& Engine::sequence(int seq) {
  const auto it = sequences_.find(seq);
  if (it == sequences_.end()) throw Error("unknown sequence " + std::to_string(seq));
  return it->second;
}

const std::vector<std::int32_t>& Engine::tokens(int seq) const {
  const auto it = sequences_.find(seq);
  if (it == sequences_.end()) throw Error("unknown sequence " + std::to_string(seq));
  return it->second.tokens;
}

void Engine::add_sequence(int seq, std::span<const std::int32_t> prompt,
                          const SamplerConfig& sampler) {
  if (prompt.empty()) throw Error("a sequence needs at least one prompt token");
  if (sequences_.contains(seq)) throw Error("sequence " + std::to_string(seq) + " already exists");
  sequences_.emplace(seq, Sequence{.row = -1,
                                   .blocks = {},
                                   .tokens = {prompt.begin(), prompt.end()},
                                   .n_cached = 0,
                                   .sampler = Sampler(sampler)});
}

void Engine::release_cache(Sequence& s) {
  for (const int block : s.blocks) blocks_.free(block);
  s.blocks.clear();
  if (s.row >= 0) rows_.free(s.row);
  s.row = -1;
  s.n_cached = 0;
}

StepResult Engine::execute(const StepPlan& plan) {
  for (const int seq : plan.free) {
    release_cache(sequence(seq));
    sequences_.erase(seq);
  }
  for (const int seq : plan.preempt) release_cache(sequence(seq));
  for (const StepPlan::Alloc& alloc : plan.alloc) {
    Sequence& s = sequence(alloc.seq);
    if (s.row < 0) s.row = rows_.allocate();
    for (int i = 0; i < alloc.n_blocks; ++i) s.blocks.push_back(blocks_.allocate());
    set_blocks(s.row, s.blocks);
  }

  StepResult result;
  if (plan.run.empty()) return result;

  const int block_size = kv_block_size();
  std::vector<StepRow> rows;
  std::vector<std::int32_t> out_rows;
  std::vector<int> emitting;      // sequence ids, parallel to out_rows
  std::vector<int> cached_after;  // per plan.run entry: the sequence's n_cached once the step ran
  bool all_greedy = true;
  for (const StepPlan::Run& run : plan.run) {
    Sequence& s = sequence(run.seq);
    const int end = s.n_cached + run.n_tokens;
    if (run.n_tokens <= 0 || end > static_cast<int>(s.tokens.size())) {
      throw Error("plan runs tokens sequence " + std::to_string(run.seq) + " does not have");
    }
    if (end > static_cast<int>(s.blocks.size()) * block_size) {
      throw Error("sequence " + std::to_string(run.seq) + " has no cache block for its tokens");
    }
    for (int pos = s.n_cached; pos < end; ++pos) {
      rows.push_back({.token_id = s.tokens[pos],
                      .position = pos,
                      .seq = s.row,
                      .kv_slot = s.blocks[pos / block_size] * block_size + pos % block_size});
    }
    cached_after.push_back(end);
    if (end == static_cast<int>(s.tokens.size())) {
      out_rows.push_back(static_cast<std::int32_t>(rows.size()) - 1);
      emitting.push_back(run.seq);
      all_greedy = all_greedy && s.sampler.config().greedy();
    }
  }

  forward(rows, out_rows, {.times = op_times_});
  // Cached counts advance only now, so a step that throws can be planned again. The plan's
  // frees and allocations above are not undone.
  for (std::size_t i = 0; i < plan.run.size(); ++i) {
    sequence(plan.run[i].seq).n_cached = cached_after[i];
  }
  const std::span<const std::int32_t> greedy = next_ids();
  const std::vector<float> host_logits = all_greedy ? std::vector<float>{} : logits();
  const std::size_t vocab = static_cast<std::size_t>(model_->config().vocab_size);
  for (std::size_t i = 0; i < emitting.size(); ++i) {
    Sequence& s = sequence(emitting[i]);
    const std::int32_t token =
        s.sampler.config().greedy()
            ? greedy[i]
            : s.sampler.sample(std::span(host_logits).subspan(i * vocab, vocab));
    s.tokens.push_back(token);
    result.tokens.push_back({emitting[i], token});
  }
  return result;
}

}  // namespace kvb::cuda
