#include "kvbound/cuda/asr_engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <optional>
#include <string>
#include <utility>

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
      rows_(config_.max_seqs),
      cross_blocks_(config_.max_seqs),
      cross_of_row_(config_.max_seqs),
      graphs_(config_.max_graphs) {
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
  for (int row = 0; row < config_.max_seqs; ++row) point_cross(row, row);

  const std::size_t encoder_rows =
      static_cast<std::size_t>(options_.max_encode_batch) * cfg.max_source_positions;
  const std::size_t max_rows = std::max<std::size_t>(config_.max_step_tokens, encoder_rows);
  buffers_ = StepBuffers(max_rows, config_.max_outputs);
  mel_host_ = PinnedBuffer(options_.max_encode_batch * feature_size() * sizeof(float));
  mel_dev_ = DeviceBuffer(mel_host_.bytes());
  KVB_CUDA(cudaStreamSynchronize(stream_));
}

AsrEngine::~AsrEngine() {
  graphs_.clear();
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

AsrBlockUsage AsrEngine::block_usage() const {
  AsrBlockUsage usage{.self_blocks = blocks_.n_blocks(),
                      .self_free = blocks_.n_free(),
                      .cross_blocks = cross_blocks_.n_blocks(),
                      .cross_free = cross_blocks_.n_free(),
                      .rows = rows_.n_blocks(),
                      .rows_free = rows_.n_free()};
  for (int block = 0; block < blocks_.n_blocks(); ++block)
    usage.self_refs += blocks_.ref_count(block);
  for (int block = 0; block < cross_blocks_.n_blocks(); ++block) {
    usage.cross_refs += cross_blocks_.ref_count(block);
  }
  return usage;
}

void AsrEngine::set_blocks(int seq, std::span<const int> blocks) {
  if (seq < 0 || seq >= config_.max_seqs) throw Error("block-table row out of range");
  self_->set_blocks(seq, blocks, stream_);
}

void AsrEngine::point_cross(int row, int block) {
  cross_->set_blocks(row, {&block, 1}, stream_);
  cross_of_row_[row] = block;
}

void AsrEngine::encode(std::span<const float> features, std::span<const int> seqs,
                       const ForwardOptions& options) {
  const std::size_t size = feature_size();
  if (features.size() != seqs.size() * size) {
    throw Error("features hold " + std::to_string(features.size()) + " values; " +
                std::to_string(seqs.size()) + " clips need " + std::to_string(size) + " each");
  }
  std::vector<std::span<const float>> clips;
  std::vector<int> blocks;
  for (std::size_t i = 0; i < seqs.size(); ++i) {
    const int seq = seqs[i];
    if (seq < 0 || seq >= config_.max_seqs || std::count(seqs.begin(), seqs.end(), seq) != 1) {
      throw Error("an encoder step needs a distinct block-table row in range for every clip");
    }
    clips.push_back(features.subspan(i * size, size));
    blocks.push_back(cross_of_row_[seq]);
  }
  encode_clips(clips, blocks, options);
}

void AsrEngine::encode_clips(std::span<const std::span<const float>> clips,
                             std::span<const int> blocks, const ForwardOptions& options) {
  const WhisperConfig& cfg = model_->config();
  const int n_audio = static_cast<int>(clips.size());
  const int n_ctx = static_cast<int>(cfg.max_source_positions);
  const std::size_t n_mels = static_cast<std::size_t>(cfg.n_mels);
  const std::size_t n_frames = static_cast<std::size_t>(cfg.n_frames());
  if (n_audio == 0 || n_audio > options_.max_encode_batch || blocks.size() != clips.size()) {
    throw Error("an encoder step takes between 1 and " + std::to_string(options_.max_encode_batch) +
                " clips, each with a Cross block");
  }

  // The model reads one row per frame; the features come with one row per mel bin.
  float* mel = mel_host_.as<float>();
  std::vector<StepRow> rows(static_cast<std::size_t>(n_audio) * n_ctx);
  for (int clip = 0; clip < n_audio; ++clip) {
    const int block = blocks[clip];
    if (block < 0 || block >= cross_blocks_.n_blocks() ||
        std::count(blocks.begin(), blocks.end(), block) != 1) {
      throw Error("an encoder step needs a distinct Cross block for every clip");
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
    for (int pos = 0; pos < n_ctx; ++pos) {
      rows[clip * n_ctx + pos] = {
          .token_id = 0, .position = pos, .seq = clip, .kv_slot = block * n_ctx + pos};
    }
  }
  const int n_tokens = n_audio * n_ctx;
  const std::vector<AttnTile> tiles = plan_attention_tiles(rows, kernels::kAttnTileRows);
  const std::size_t n_values = n_audio * n_mels * n_frames;
  KVB_CUDA(cudaMemcpyAsync(mel_dev_.as(), mel, n_values * sizeof(float), cudaMemcpyHostToDevice,
                           stream_));
  buffers_.upload_rows(rows, stream_);
  buffers_.upload_tiles(tiles, stream_);
  kernels::f32_to_f16(mel_dev_.as<float>(), model_->mel(), n_values, stream_);

  const int n_tiles = static_cast<int>(tiles.size());
  if (options.capture != nullptr || options.times != nullptr) {
    const RunHooks hooks = step_hooks(model_->programs().encoder, n_tokens, 0, stream_,
                                      options.capture, options.times);
    model_->encode(buffers_.rows(), n_audio, buffers_.tiles(), n_tiles, *cross_, stream_, &hooks);
  } else {
    model_->encode(buffers_.rows(), n_audio, buffers_.tiles(), n_tiles, *cross_, stream_);
  }
  KVB_CUDA(cudaStreamSynchronize(stream_));
}

void AsrEngine::forward(std::span<const StepRow> rows, std::span<const std::int32_t> out_rows,
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

  // A step with a run of several rows of one sequence runs self attention as tiles of those runs.
  int n_tiles = 0;
  if (config_.prefill_attention) {
    const std::vector<AttnTile> tiles = plan_attention_tiles(rows, kernels::kAttnTileRows);
    if (tiles.size() < rows.size()) {
      n_tiles = static_cast<int>(tiles.size());
      buffers_.upload_tiles(tiles, stream_);
    }
  }

  const bool hooked = options.capture != nullptr || options.times != nullptr;
  const RunHooks hooks = step_hooks(model_->programs().decoder, n_tokens, n_out, stream_,
                                    options.capture, options.times);
  // Decode steps replay a graph. The suppress rules read each row's position from the step's
  // rows, so one graph serves the first generated token too.
  graphs_.step(
      config_.use_graph, n_tokens, n_out, n_tiles, config_.max_seqs, hooked, stream_,
      [&](bool graphed) {
        model_->run(buffers_.rows(), buffers_.out_rows(), n_tokens, n_out,
                    graphed ? nullptr : buffers_.tiles(), graphed ? 0 : n_tiles, *self_, *cross_,
                    stream_, hooked ? &hooks : nullptr, !options.raw_logits);
      },
      options.raw_logits ? 1 : 0);

  buffers_.fetch_next_ids(model_->next_ids(), static_cast<std::size_t>(n_out), stream_);
  KVB_CUDA(cudaStreamSynchronize(stream_));
}

std::vector<float> AsrEngine::logits() {
  std::vector<float> host(next_ids().size() *
                          static_cast<std::size_t>(model_->config().vocab_size));
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
  sequences_.emplace(seq, Sequence{CachedSequence{.tokens = model_->rules().prompt}, -1, false,
                                   std::vector<float>(features.begin(), features.end())});
}

void AsrEngine::release_cache(Sequence& s) {
  if (s.cross_block >= 0) {
    cross_blocks_.free(s.cross_block);
    point_cross(s.row, s.row);
    s.cross_block = -1;
  }
  kvb::release_cache(s, blocks_, rows_);
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
    if (s.row < 0) {
      s.row = rows_.allocate();
      s.cross_block = cross_blocks_.allocate();
      point_cross(s.row, s.cross_block);
    }
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
    std::vector<int> cross_blocks;
    for (const int seq : std::span(plan.encode).subspan(first, n)) {
      const Sequence& s = sequence(seq);
      if (s.row < 0) throw Error("sequence " + std::to_string(seq) + " has no row to encode into");
      clips.push_back(s.features);
      cross_blocks.push_back(s.cross_block);
    }
    encode_clips(clips, cross_blocks, {.times = encoder_times_});
    for (const int seq : std::span(plan.encode).subspan(first, n)) sequence(seq).encoded = true;
  }
  if (!plan.encode.empty()) {
    encode_seconds_ =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - encode_start).count();
  }

  StepResult result;
  if (plan.run.empty()) return result;

  for (const StepPlan::Run& run : plan.run) {
    if (!sequence(run.seq).encoded) {
      throw Error("plan runs sequence " + std::to_string(run.seq) + " before encoding its clip");
    }
  }
  const StepRows step = build_step_rows(
      plan.run, kv_block_size(), [&](int seq) -> const CachedSequence& { return sequence(seq); });

  forward(step.rows, step.out_rows, {.times = decoder_times_});
  // Cached counts advance only now, so a step that throws can be planned again. The plan's
  // frees, allocations and encoder steps above are not undone.
  for (std::size_t i = 0; i < plan.run.size(); ++i) {
    sequence(plan.run[i].seq).n_cached = step.cached_after[i];
  }
  const std::span<const std::int32_t> greedy = next_ids();
  for (std::size_t i = 0; i < step.emitting.size(); ++i) {
    sequence(step.emitting[i]).tokens.push_back(greedy[i]);
    result.tokens.push_back({step.emitting[i], greedy[i]});
  }
  return result;
}

void AsrEngine::beam_candidates_on_host(std::span<const std::int32_t> suppressed, int k,
                                        std::vector<BeamCandidate>& out) {
  const std::vector<float> host = logits();
  const std::size_t vocab = static_cast<std::size_t>(model_->config().vocab_size);
  for (std::size_t first = 0; first < host.size(); first += vocab) {
    append_beam_candidates({host.data() + first, vocab}, suppressed, k, out);
  }
}

BeamTranscript AsrEngine::beam_transcribe(std::span<const float> features,
                                          const BeamOptions& options) {
  using Clock = std::chrono::steady_clock;
  const Clock::time_point start = Clock::now();
  const auto elapsed = [&] { return std::chrono::duration<double>(Clock::now() - start).count(); };
  if (!sequences_.empty()) throw Error("beam_transcribe() needs an engine with no sequences");
  if (features.size() != feature_size()) throw Error("a clip's features have the wrong size");
  const WhisperRules& rules = model_->rules();
  const int n_prompt = static_cast<int>(rules.prompt.size());
  const int max_length =
      static_cast<int>(std::min<std::int64_t>(rules.max_length, config_.max_ctx));
  if (max_length <= n_prompt) throw Error("max_ctx leaves no room after the prompt");
  const int num_beams = options.num_beams;
  if (num_beams <= 0 || num_beams > config_.max_seqs) {
    throw Error("num_beams must be between 1 and max_seqs");
  }
  const int block_size = kv_block_size();
  // Every beam may hold a whole sequence; forking in block_allocator.h's order needs no more.
  const int needed = num_beams * ((max_length + block_size - 1) / block_size);
  if (needed > blocks_.n_free()) {
    throw Error(std::to_string(num_beams) + " beams need " + std::to_string(needed) +
                " Self blocks; the pool has " + std::to_string(blocks_.n_free()));
  }
  BeamSearch search(
      {.num_beams = num_beams,
       .eos_token_id = options.eos_token_id >= 0 ? options.eos_token_id : rules.eos_token_id,
       .max_new_tokens = max_length - n_prompt,
       .length_penalty = options.length_penalty});
  // The first generated token is chosen under both suppress lists.
  std::vector<std::int32_t> first_suppressed = rules.suppress_tokens;
  first_suppressed.insert(first_suppressed.end(), rules.begin_suppress_tokens.begin(),
                          rules.begin_suppress_tokens.end());

  // What the search holds, released on every way out: block-table rows, the Cross block with
  // one reference per beam that points at it, and each beam's Self blocks. `next` holds the
  // beams of the next step while a fork builds them.
  std::vector<int> rows;
  int cross = -1;
  int cross_refs = 0;
  std::vector<std::vector<int>> beams, next;
  const auto release = [&] {
    for (std::vector<int>& blocks : beams) release_blocks(blocks_, blocks);
    for (std::vector<int>& blocks : next) release_blocks(blocks_, blocks);
    for (; cross_refs > 0; --cross_refs) cross_blocks_.free(cross);
    for (const int row : rows) {
      point_cross(row, row);
      rows_.free(row);
    }
    rows.clear();
  };

  BeamTranscript result;
  try {
    for (int i = 0; i < num_beams; ++i) rows.push_back(rows_.allocate());
    cross = cross_blocks_.allocate();
    cross_refs = 1;  // the prompt beam's
    for (const int row : rows) point_cross(row, cross);
    const Clock::time_point encode_start = Clock::now();
    encode_clips({&features, 1}, {&cross, 1}, {.times = encoder_times_});
    result.encode_s = std::chrono::duration<double>(Clock::now() - encode_start).count();

    // The prompt beam takes a block at each block boundary of the prompt.
    beams.assign(1, {});
    for (int pos = 0; pos < n_prompt; ++pos) prepare_write(blocks_, beams[0], pos, block_size);
    set_blocks(rows[0], beams[0]);
    const auto slot = [&](const std::vector<int>& blocks, int pos) {
      return blocks[pos / block_size] * block_size + pos % block_size;
    };
    std::vector<StepRow> step_rows;
    for (int pos = 0; pos < n_prompt; ++pos) {
      step_rows.push_back({.token_id = rules.prompt[pos],
                           .position = pos,
                           .seq = rows[0],
                           .kv_slot = slot(beams[0], pos)});
    }
    std::vector<std::int32_t> out_rows{n_prompt - 1};
    std::vector<BeamCandidate> candidates;
    std::vector<std::pair<BlockCopy, int>> copies;  // and the positions to copy
    while (true) {
      forward(step_rows, out_rows, {.times = decoder_times_, .raw_logits = true});
      // Every live beam now caches positions [0, n).
      const int n = step_rows.back().position + 1;
      candidates.clear();
      beam_candidates_on_host(n == n_prompt ? first_suppressed : rules.suppress_tokens,
                              search.n_candidates(), candidates);
      const BeamStep& step = search.step(candidates);
      result.step_s.push_back(elapsed());
      if (step.done) break;

      // Fork: each next beam shares its parent's blocks, then the old beams let go of theirs.
      next.clear();
      for (const Beam& beam : step.beams) {
        next.push_back(fork_blocks(blocks_, beams[beam.parent], n, block_size));
        cross_blocks_.retain(cross);
        ++cross_refs;
      }
      for (std::vector<int>& blocks : beams) {
        release_blocks(blocks_, blocks);
        cross_blocks_.free(cross);
        --cross_refs;
      }
      beams.swap(next);
      next.clear();

      // Every beam writes position n; without sharing, every block it holds is made its own. A
      // copy covers the positions below n that its block holds, all the block has written.
      copies.clear();
      const auto make_writable = [&](std::vector<int>& blocks, int position) {
        if (const std::optional<BlockCopy> copy =
                prepare_write(blocks_, blocks, position, block_size)) {
          const int first = position / block_size * block_size;
          copies.push_back({*copy, std::min(block_size, n - first)});
        }
      };
      for (std::size_t i = 0; i < beams.size(); ++i) {
        std::vector<int>& blocks = beams[i];
        if (!options_.beam_share_blocks) {
          const int held = static_cast<int>(blocks.size());
          for (int b = 0; b < held; ++b) make_writable(blocks, b * block_size);
        }
        make_writable(blocks, n);
        set_blocks(rows[i], blocks);
      }
      // Queued before the step on the same stream, so each copy reads its source before any
      // beam writes position n.
      for (const auto& [copy, positions] : copies) {
        self_->copy_block(copy.src, copy.dst, stream_, positions);
      }
      result.block_copies += static_cast<int>(copies.size());

      step_rows.clear();
      out_rows.clear();
      for (std::size_t i = 0; i < beams.size(); ++i) {
        step_rows.push_back({.token_id = step.beams[i].token,
                             .position = n,
                             .seq = rows[i],
                             .kv_slot = slot(beams[i], n)});
        out_rows.push_back(static_cast<std::int32_t>(i));
      }
    }
    const BeamHypothesis& best = search.best();
    result.ids = best.tokens;
    result.score = best.score;
    result.hypotheses.assign(search.hypotheses().begin(), search.hypotheses().end());
  } catch (...) {
    release();
    throw;
  }
  release();
  result.seconds = elapsed();
  return result;
}

}  // namespace kvb::cuda
