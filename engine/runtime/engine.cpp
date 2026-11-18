#include "kvbound/cuda/engine.h"

#include <numeric>

#include "kvbound/cuda/kernels.h"

namespace kvb::cuda {

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

Engine::Engine(const std::filesystem::path& model_dir, const EngineConfig& config)
    : config_(resolved(config)),
      stream_(create_stream()),
      blocks_(config_.kv_blocks),
      rows_(config_.max_seqs),
      buffers_(config_.max_step_tokens, config_.max_outputs),
      graphs_(config_.max_graphs) {
  model_ = std::make_unique<LlamaModel>(
      model_dir,
      LlamaModelConfig{.max_ctx = config_.max_ctx,
                       .max_step_tokens = config_.max_step_tokens,
                       .max_outputs = config_.max_outputs,
                       .plan = config_.plan,
                       .gemm = config_.gemm,
                       .fuse = config_.fuse,
                       .linear = {.f16_gemv_rows = config_.f16_gemv_rows,
                                  .quantized_gemm_rows = config_.quantized_gemm_rows,
                                  .expand_quantized = config_.expand_quantized},
                       .wide_key_loads = config_.wide_key_loads,
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
  graphs_.clear();
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
  check_step(rows, out_rows,
             {.max_step_tokens = config_.max_step_tokens,
              .max_outputs = config_.max_outputs,
              .max_ctx = config_.max_ctx,
              .max_seqs = config_.max_seqs,
              .vocab_size = model_->config().vocab_size});
  const int n_tokens = static_cast<int>(rows.size());
  const int n_out = static_cast<int>(out_rows.size());
  buffers_.upload_rows(rows, stream_);
  buffers_.upload_out_rows(out_rows, stream_);

  // A step with a run of several rows of one sequence runs attention as tiles of those runs.
  int n_tiles = 0;
  if (config_.prefill_attention) {
    const std::vector<AttnTile> tiles = plan_attention_tiles(rows, kernels::kAttnTileRows);
    if (tiles.size() < rows.size()) {
      n_tiles = static_cast<int>(tiles.size());
      buffers_.upload_tiles(tiles, stream_);
    }
  }

  const bool hooked = options.capture != nullptr || options.times != nullptr;
  const RunHooks hooks =
      step_hooks(model_->program(), n_tokens, n_out, stream_, options.capture, options.times);
  // Decode steps, one row per sequence, replay a graph.
  graphs_.step(config_.use_graph, n_tokens, n_out, n_tiles, config_.max_seqs, hooked, stream_,
               [&](bool graphed) {
                 model_->run(buffers_.rows(), buffers_.out_rows(), n_tokens, n_out,
                             graphed ? nullptr : buffers_.tiles(), graphed ? 0 : n_tiles, *kv_,
                             stream_, hooked ? &hooks : nullptr);
               });

  buffers_.fetch_next_ids(model_->next_ids(), static_cast<std::size_t>(n_out), stream_);
  KVB_CUDA(cudaStreamSynchronize(stream_));
}

std::vector<float> Engine::logits() {
  std::vector<float> host(next_ids().size() *
                          static_cast<std::size_t>(model_->config().vocab_size));
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
  Sequence s{CachedSequence{.tokens = {prompt.begin(), prompt.end()}}, Sampler(sampler)};
  sequences_.emplace(seq, std::move(s));
}

StepResult Engine::execute(const StepPlan& plan) {
  for (const int seq : plan.free) {
    release_cache(sequence(seq), blocks_, rows_);
    sequences_.erase(seq);
  }
  for (const int seq : plan.preempt) release_cache(sequence(seq), blocks_, rows_);
  for (const StepPlan::Alloc& alloc : plan.alloc) {
    Sequence& s = sequence(alloc.seq);
    if (s.row < 0) s.row = rows_.allocate();
    for (int i = 0; i < alloc.n_blocks; ++i) s.blocks.push_back(blocks_.allocate());
    set_blocks(s.row, s.blocks);
  }

  StepResult result;
  if (plan.run.empty()) return result;

  const StepRows step = build_step_rows(
      plan.run, kv_block_size(), [&](int seq) -> const CachedSequence& { return sequence(seq); });
  bool all_greedy = true;
  for (const int seq : step.emitting) {
    all_greedy = all_greedy && sequence(seq).sampler.config().greedy();
  }

  forward(step.rows, step.out_rows, {.times = op_times_});
  // Cached counts advance only now, so a step that throws can be planned again. The plan's
  // frees and allocations above are not undone.
  for (std::size_t i = 0; i < plan.run.size(); ++i) {
    sequence(plan.run[i].seq).n_cached = step.cached_after[i];
  }
  const std::span<const std::int32_t> greedy = next_ids();
  const std::vector<float> host_logits = all_greedy ? std::vector<float>{} : logits();
  const std::size_t vocab = static_cast<std::size_t>(model_->config().vocab_size);
  for (std::size_t i = 0; i < step.emitting.size(); ++i) {
    Sequence& s = sequence(step.emitting[i]);
    const std::int32_t token =
        s.sampler.config().greedy()
            ? greedy[i]
            : s.sampler.sample(std::span(host_logits).subspan(i * vocab, vocab));
    s.tokens.push_back(token);
    result.tokens.push_back({step.emitting[i], token});
  }
  return result;
}

}  // namespace kvb::cuda
