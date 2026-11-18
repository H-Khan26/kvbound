#include "kvbound/cuda/asr_engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <string>

#include "kvbound/cuda/kernels.h"

namespace kvb::cuda {
namespace {

// Largest magnitude FP16 holds: a feature beyond it would round to infinity.
constexpr float kMaxHalf = 65504.0f;

EngineConfig resolved_for(const WhisperConfig& cfg, EngineConfig config,
                          const AsrOptions& options) {
  const int max_target = static_cast<int>(cfg.max_target_positions);
  if (config.max_ctx <= 0) config.max_ctx = max_target;
  if (config.max_ctx > max_target) {
    throw Error("max_ctx exceeds the model's " + std::to_string(max_target) + " target positions");
  }
  config = resolved(config);
  // Every step runs eagerly.
  config.use_graph = false;
  if (options.max_encode_batch <= 0 || options.max_encode_batch > config.max_seqs) {
    throw Error("max_encode_batch must be between 1 and max_seqs");
  }
  return config;
}

}  // namespace

AsrEngine::AsrEngine(const std::filesystem::path& model_dir, const EngineConfig& config,
                     const AsrOptions& options)
    : config_(resolved_for(load_whisper_config(model_dir / "config.json"), config, options)),
      options_(options),
      stream_(create_stream()),
      blocks_(config_.kv_blocks),
      rows_(config_.max_seqs) {
  model_ = std::make_unique<WhisperModel>(
      model_dir,
      WhisperModelConfig{.max_ctx = config_.max_ctx,
                         .max_step_tokens = config_.max_step_tokens,
                         .max_outputs = config_.max_outputs,
                         .max_encode_batch = options_.max_encode_batch,
                         .plan = config_.plan,
                         .gemm = config_.gemm,
                         .fuse = config_.fuse,
                         .linear = {.f16_gemv_rows = config_.f16_gemv_rows,
                                    .quantized_gemm_rows = config_.quantized_gemm_rows,
                                    .expand_quantized = config_.expand_quantized},
                         .wide_key_loads = config_.wide_key_loads,
                         .dequant_slice_bytes = config_.dequant_slice_bytes},
      stream_);
  const WhisperConfig& cfg = model_->config();
  const std::int64_t head_dim = cfg.d_model / cfg.decoder_heads;
  const int blocks_per_seq = (config_.max_ctx + config_.kv_block_size - 1) / config_.kv_block_size;
  self_ = std::make_unique<KvPool>(KvDims{.n_layers = cfg.decoder_layers,
                                          .n_blocks = config_.kv_blocks,
                                          .n_kv_heads = cfg.decoder_heads,
                                          .block_size = config_.kv_block_size,
                                          .head_dim = head_dim},
                                   config_.kv_layout, config_.max_seqs, blocks_per_seq,
                                   config_.kv_dtype);
  // One head's keys of one clip are a contiguous stream, which every decoder row reads whole.
  cross_ =
      std::make_unique<KvPool>(KvDims{.n_layers = cfg.decoder_layers,
                                      .n_blocks = config_.max_seqs,
                                      .n_kv_heads = cfg.decoder_heads,
                                      .block_size = cfg.max_source_positions,
                                      .head_dim = head_dim},
                               KvLayout::HeadMajor, config_.max_seqs, 1, options_.cross_kv_dtype);
  for (int row = 0; row < config_.max_seqs; ++row) cross_->set_blocks(row, {&row, 1}, stream_);

  const std::size_t encoder_rows =
      static_cast<std::size_t>(options_.max_encode_batch) * cfg.max_source_positions;
  const std::size_t max_rows = std::max<std::size_t>(config_.max_step_tokens, encoder_rows);
  rows_host_ = PinnedBuffer(max_rows * sizeof(StepRow));
  out_rows_host_ = PinnedBuffer(config_.max_step_tokens * sizeof(std::int32_t));
  tiles_host_ = PinnedBuffer(max_rows * sizeof(AttnTile));
  next_ids_ = PinnedBuffer(config_.max_step_tokens * sizeof(std::int32_t));
  mel_host_ = PinnedBuffer(options_.max_encode_batch * feature_size() * sizeof(float));
  rows_dev_ = DeviceBuffer(rows_host_.bytes());
  out_rows_dev_ = DeviceBuffer(out_rows_host_.bytes());
  tiles_dev_ = DeviceBuffer(tiles_host_.bytes());
  mel_dev_ = DeviceBuffer(mel_host_.bytes());
  KVB_CUDA(cudaStreamSynchronize(stream_));
}

AsrEngine::~AsrEngine() {
  model_.reset();
  self_.reset();
  cross_.reset();
  if (stream_ != nullptr) cudaStreamDestroy(stream_);
}

std::size_t AsrEngine::feature_size() const noexcept {
  const WhisperConfig& cfg = model_->config();
  return static_cast<std::size_t>(cfg.n_mels * cfg.n_frames());
}

AsrMemory AsrEngine::memory() const {
  return AsrMemory{
      .weights = model_->weight_bytes(),
      .self_kv = self_->bytes(),
      .cross_kv = cross_->bytes(),
      .arena = model_->arena_bytes(),
      .encoder_arena = model_->encoder_plan().arena_bytes,
      .decoder_arena = model_->decoder_plan().arena_bytes,
      .dequant = model_->dequant_bytes(),
  };
}

void AsrEngine::set_blocks(int seq, std::span<const int> blocks) {
  if (seq < 0 || seq >= config_.max_seqs) throw Error("block-table row out of range");
  self_->set_blocks(seq, blocks, stream_);
}

void AsrEngine::encode(std::span<const float> features, std::span<const int> seqs,
                       const ForwardOptions& options) {
  const std::size_t size = feature_size();
  if (features.size() != seqs.size() * size) {
    throw Error("features hold " + std::to_string(features.size()) + " values; " +
                std::to_string(seqs.size()) + " clips need " + std::to_string(size) + " each");
  }
  std::vector<std::span<const float>> clips;
  for (std::size_t i = 0; i < seqs.size(); ++i) clips.push_back(features.subspan(i * size, size));
  encode_clips(clips, seqs, options);
}

void AsrEngine::encode_clips(std::span<const std::span<const float>> clips,
                             std::span<const int> seqs, const ForwardOptions& options) {
  const WhisperConfig& cfg = model_->config();
  const int n_audio = static_cast<int>(clips.size());
  const int n_ctx = static_cast<int>(cfg.max_source_positions);
  const std::size_t n_mels = static_cast<std::size_t>(cfg.n_mels);
  const std::size_t n_frames = static_cast<std::size_t>(cfg.n_frames());
  if (n_audio == 0 || n_audio > options_.max_encode_batch || seqs.size() != clips.size()) {
    throw Error("an encoder step takes between 1 and " + std::to_string(options_.max_encode_batch) +
                " clips, each with a block-table row");
  }

  // The model reads one row per frame; the features come with one row per mel bin.
  float* mel = mel_host_.as<float>();
  StepRow* rows = rows_host_.as<StepRow>();
  for (int clip = 0; clip < n_audio; ++clip) {
    const int seq = seqs[clip];
    if (seq < 0 || seq >= config_.max_seqs || std::count(seqs.begin(), seqs.end(), seq) != 1) {
      throw Error("an encoder step needs a distinct block-table row in range for every clip");
    }
    const std::span<const float> source = clips[clip];
    if (source.size() != n_mels * n_frames) throw Error("a clip's features have the wrong size");
    float* frames = mel + clip * n_mels * n_frames;
    for (std::size_t bin = 0; bin < n_mels; ++bin) {
      for (std::size_t frame = 0; frame < n_frames; ++frame) {
        const float value = source[bin * n_frames + frame];
        if (!(std::abs(value) <= kMaxHalf)) throw Error("features must be finite in FP16");
        frames[frame * n_mels + bin] = value;
      }
    }
    // Row r of the Cross cache owns block r.
    for (int pos = 0; pos < n_ctx; ++pos) {
      rows[clip * n_ctx + pos] = {
          .token_id = 0, .position = pos, .seq = clip, .kv_slot = seq * n_ctx + pos};
    }
  }
  const int n_tokens = n_audio * n_ctx;
  const std::vector<AttnTile> tiles =
      plan_attention_tiles({rows, static_cast<std::size_t>(n_tokens)}, kernels::kAttnTileRows);
  std::memcpy(tiles_host_.as(), tiles.data(), tiles.size() * sizeof(AttnTile));

  const std::size_t n_values = n_audio * n_mels * n_frames;
  KVB_CUDA(cudaMemcpyAsync(mel_dev_.as(), mel, n_values * sizeof(float), cudaMemcpyHostToDevice,
                           stream_));
  KVB_CUDA(cudaMemcpyAsync(rows_dev_.as(), rows, n_tokens * sizeof(StepRow), cudaMemcpyHostToDevice,
                           stream_));
  KVB_CUDA(cudaMemcpyAsync(tiles_dev_.as(), tiles_host_.as(), tiles.size() * sizeof(AttnTile),
                           cudaMemcpyHostToDevice, stream_));
  kernels::f32_to_f16(mel_dev_.as<float>(), model_->mel(), n_values, stream_);

  const int n_tiles = static_cast<int>(tiles.size());
  if (options.capture != nullptr || options.times != nullptr) {
    const RunHooks hooks = step_hooks(model_->programs().encoder, n_tokens, 0, stream_,
                                      options.capture, options.times);
    model_->encode(rows_dev_.as<StepRow>(), n_audio, tiles_dev_.as<AttnTile>(), n_tiles, *cross_,
                   stream_, &hooks);
  } else {
    model_->encode(rows_dev_.as<StepRow>(), n_audio, tiles_dev_.as<AttnTile>(), n_tiles, *cross_,
                   stream_);
  }
  KVB_CUDA(cudaStreamSynchronize(stream_));
}

void AsrEngine::forward(std::span<const StepRow> rows, std::span<const std::int32_t> out_rows,
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
  for (const std::int32_t row : out_rows) {
    if (row < 0 || row >= n_tokens) throw Error("output row is out of range");
  }

  std::memcpy(rows_host_.as(), rows.data(), rows.size_bytes());
  std::memcpy(out_rows_host_.as(), out_rows.data(), out_rows.size_bytes());
  KVB_CUDA(cudaMemcpyAsync(rows_dev_.as(), rows_host_.as(), rows.size_bytes(),
                           cudaMemcpyHostToDevice, stream_));
  if (n_out > 0) {
    KVB_CUDA(cudaMemcpyAsync(out_rows_dev_.as(), out_rows_host_.as(), out_rows.size_bytes(),
                             cudaMemcpyHostToDevice, stream_));
  }

  // A step with a run of several rows of one sequence runs self attention as tiles of those runs.
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

  if (options.capture != nullptr || options.times != nullptr) {
    const RunHooks hooks = step_hooks(model_->programs().decoder, n_tokens, n_out, stream_,
                                      options.capture, options.times);
    model_->run(rows_dev_.as<StepRow>(), out_rows_dev_.as<std::int32_t>(), n_tokens, n_out,
                tiles_dev_.as<AttnTile>(), n_tiles, *self_, *cross_, stream_, &hooks);
  } else {
    model_->run(rows_dev_.as<StepRow>(), out_rows_dev_.as<std::int32_t>(), n_tokens, n_out,
                tiles_dev_.as<AttnTile>(), n_tiles, *self_, *cross_, stream_);
  }

  if (n_out > 0) {
    KVB_CUDA(cudaMemcpyAsync(next_ids_.as(), model_->next_ids(), n_out * sizeof(std::int32_t),
                             cudaMemcpyDeviceToHost, stream_));
  }
  KVB_CUDA(cudaStreamSynchronize(stream_));
  last_n_out_ = static_cast<std::size_t>(n_out);
}

std::vector<float> AsrEngine::logits() {
  std::vector<float> host(last_n_out_ * static_cast<std::size_t>(model_->config().vocab_size));
  if (!host.empty()) {
    KVB_CUDA(cudaMemcpy(host.data(), model_->logits(), host.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  }
  return host;
}

AsrEngine::Sequence& AsrEngine::sequence(int seq) {
  const auto it = sequences_.find(seq);
  if (it == sequences_.end()) throw Error("unknown sequence " + std::to_string(seq));
  return it->second;
}

const std::vector<std::int32_t>& AsrEngine::tokens(int seq) const {
  const auto it = sequences_.find(seq);
  if (it == sequences_.end()) throw Error("unknown sequence " + std::to_string(seq));
  return it->second.tokens;
}

void AsrEngine::add_sequence(int seq, std::span<const float> features) {
  if (features.size() != feature_size()) throw Error("a clip's features have the wrong size");
  if (sequences_.contains(seq)) throw Error("sequence " + std::to_string(seq) + " already exists");
  sequences_.emplace(seq, Sequence{.tokens = model_->rules().prompt,
                                   .features = {features.begin(), features.end()}});
}

void AsrEngine::release_cache(Sequence& s) {
  for (const int block : s.blocks) blocks_.free(block);
  s.blocks.clear();
  if (s.row >= 0) rows_.free(s.row);
  s.row = -1;
  s.n_cached = 0;
  s.encoded = false;
}

StepResult AsrEngine::execute(const StepPlan& plan) {
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

  // The clips of plan.encode, a step's worth at a time.
  encode_seconds_ = 0.0;
  const auto encode_start = std::chrono::steady_clock::now();
  for (std::size_t first = 0; first < plan.encode.size(); first += options_.max_encode_batch) {
    const std::size_t n =
        std::min<std::size_t>(options_.max_encode_batch, plan.encode.size() - first);
    std::vector<std::span<const float>> clips;
    std::vector<int> table_rows;
    for (const int seq : std::span(plan.encode).subspan(first, n)) {
      const Sequence& s = sequence(seq);
      if (s.row < 0) throw Error("sequence " + std::to_string(seq) + " has no row to encode into");
      clips.push_back(s.features);
      table_rows.push_back(s.row);
    }
    encode_clips(clips, table_rows, {.times = encoder_times_});
    for (const int seq : std::span(plan.encode).subspan(first, n)) sequence(seq).encoded = true;
  }
  if (!plan.encode.empty()) {
    encode_seconds_ =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - encode_start).count();
  }

  StepResult result;
  if (plan.run.empty()) return result;

  const int block_size = kv_block_size();
  std::vector<StepRow> rows;
  std::vector<std::int32_t> out_rows;
  std::vector<int> emitting;      // sequence ids, parallel to out_rows
  std::vector<int> cached_after;  // per plan.run entry: the sequence's n_cached once the step ran
  for (const StepPlan::Run& run : plan.run) {
    Sequence& s = sequence(run.seq);
    const int end = s.n_cached + run.n_tokens;
    if (run.n_tokens <= 0 || end > static_cast<int>(s.tokens.size())) {
      throw Error("plan runs tokens sequence " + std::to_string(run.seq) + " does not have");
    }
    if (end > static_cast<int>(s.blocks.size()) * block_size) {
      throw Error("sequence " + std::to_string(run.seq) + " has no cache block for its tokens");
    }
    if (!s.encoded) {
      throw Error("plan runs sequence " + std::to_string(run.seq) + " before encoding its clip");
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
    }
  }

  forward(rows, out_rows, {.times = decoder_times_});
  // Cached counts advance only now, so a step that throws can be planned again. The plan's
  // frees, allocations and encoder steps above are not undone.
  for (std::size_t i = 0; i < plan.run.size(); ++i) {
    sequence(plan.run[i].seq).n_cached = cached_after[i];
  }
  const std::span<const std::int32_t> greedy = next_ids();
  for (std::size_t i = 0; i < emitting.size(); ++i) {
    sequence(emitting[i]).tokens.push_back(greedy[i]);
    result.tokens.push_back({emitting[i], greedy[i]});
  }
  return result;
}

}  // namespace kvb::cuda
