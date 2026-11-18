#include "kvbound/cuda/whisper_model.h"

#include <algorithm>
#include <chrono>
#include <numeric>
#include <string>

#include "kvbound/cuda/kernels.h"

namespace kvb::cuda {
namespace {

WhisperModelConfig checked(const WhisperModelConfig& limits, const WhisperConfig& cfg) {
  const std::int64_t encoder_head_dim = cfg.d_model / cfg.encoder_heads;
  const int decoder_head_dim = static_cast<int>(cfg.d_model / cfg.decoder_heads);
  if (cfg.d_model % cfg.encoder_heads != 0 || cfg.d_model % cfg.decoder_heads != 0) {
    throw Error("d_model must be a multiple of the attention heads");
  }
  // The encoder attends on tensor cores, straight from its q|k|v product.
  if (encoder_head_dim != 64 && encoder_head_dim != 128) {
    throw Error("the encoder's attention needs head_dim 64 or 128");
  }
  const int max_positions = kernels::attention_max_positions(decoder_head_dim);
  const int max_ctx =
      static_cast<int>(std::min<std::int64_t>(cfg.max_target_positions, max_positions));
  if (limits.max_ctx <= 0 || limits.max_ctx > max_ctx) {
    throw Error("max_ctx must be between 1 and " + std::to_string(max_ctx));
  }
  if (cfg.max_source_positions > max_positions) {
    throw Error("cross attention supports about 11,000 encoder positions");
  }
  if (limits.max_encode_batch <= 0 || limits.max_step_tokens <= 0 || limits.max_outputs <= 0) {
    throw Error("max_encode_batch, max_step_tokens and max_outputs must be > 0");
  }
  return limits;
}

// Bytes of scratch for expanding quantized matrices: the largest one a step of either program
// may multiply with more rows than the in-place kernel takes, up to one slice. None unless the
// model expands.
std::size_t dequant_scratch_bytes(const Program& program, const DeviceWeights& weights,
                                  const WhisperModelConfig& limits, std::int64_t max_tokens,
                                  std::int64_t max_outputs) {
  if (!limits.linear.expand_quantized) return 0;
  std::size_t bytes = 0;
  for (const Op& op : program.ops) {
    if (op.kind != OpKind::Linear || weights.ref(op.weight).scheme == WeightScheme::F16) continue;
    const Value& out = program.values[op.out];
    if (row_count(out.rows, max_tokens, max_outputs) <= limits.linear.quantized_gemm_rows) continue;
    bytes = std::max(bytes, expand_scratch_bytes(program.values[op.in0].cols, out.cols,
                                                 limits.dequant_slice_bytes));
  }
  return bytes;
}

DeviceBuffer upload(const std::vector<std::int32_t>& values) {
  DeviceBuffer buffer(values.size() * sizeof(std::int32_t));
  if (!values.empty()) {
    KVB_CUDA(cudaMemcpy(buffer.as(), values.data(), buffer.bytes(), cudaMemcpyHostToDevice));
  }
  return buffer;
}

}  // namespace

WhisperModel::WhisperModel(const std::filesystem::path& model_dir, const WhisperModelConfig& limits,
                           cudaStream_t stream)
    : config_(load_whisper_config(model_dir / "config.json")),
      rules_(load_whisper_rules(model_dir / "generation.json")),
      limits_(checked(limits, config_)),
      programs_(build_whisper_programs(config_, rules_, limits.fuse)),
      weights_(load_whisper_weights(model_dir, programs_, config_, stream)),
      suppress_always_(upload(programs_.decoder.suppress.always)),
      suppress_begin_(upload(programs_.decoder.suppress.begin)),
      blas_(stream, limits.gemm) {
  const std::int64_t encoder_tokens = limits.max_encode_batch * config_.max_source_positions;
  const auto planned = [&](const Program& program, std::int64_t max_tokens,
                           std::int64_t max_outputs) {
    Planned p;
    p.bytes = value_bytes(program, max_tokens, max_outputs);
    p.plan = plan_memory(p.bytes, value_lifetimes(program), limits.plan);
    return p;
  };
  encoder_ = planned(programs_.encoder, encoder_tokens, 0);
  decoder_ = planned(programs_.decoder, limits.max_step_tokens, limits.max_outputs);
  arena_ = DeviceBuffer(std::max(encoder_.plan.arena_bytes, decoder_.plan.arena_bytes));
  dequant_ = DeviceBuffer(
      std::max(dequant_scratch_bytes(programs_.encoder, weights_, limits_, encoder_tokens, 0),
               dequant_scratch_bytes(programs_.decoder, weights_, limits_, limits.max_step_tokens,
                                     limits.max_outputs)));
  std::vector<std::int32_t> clips(limits.max_encode_batch);
  std::iota(clips.begin(), clips.end(), 0);
  clip_blocks_ = upload(clips);
}

void WhisperModel::encode(const StepRow* rows, int n_audio, const AttnTile* tiles, int n_tiles,
                          const KvPool& cross, cudaStream_t stream, const RunHooks* hooks) {
  if (n_audio <= 0 || n_audio > limits_.max_encode_batch) {
    throw Error("an encoder step takes between 1 and " + std::to_string(limits_.max_encode_batch) +
                " clips");
  }
  const int n_tokens = n_audio * static_cast<int>(config_.max_source_positions);
  execute(programs_.encoder, encoder_,
          {.rows = rows, .n_tokens = n_tokens, .tiles = tiles, .n_tiles = n_tiles, .cross = &cross},
          stream, hooks);
}

void WhisperModel::run(const StepRow* rows, const std::int32_t* out_rows, int n_tokens, int n_out,
                       const AttnTile* tiles, int n_tiles, const KvPool& self, const KvPool& cross,
                       cudaStream_t stream, const RunHooks* hooks) {
  execute(programs_.decoder, decoder_,
          {.rows = rows,
           .out_rows = out_rows,
           .n_tokens = n_tokens,
           .n_out = n_out,
           .tiles = tiles,
           .n_tiles = n_tiles,
           .self = &self,
           .cross = &cross},
          stream, hooks);
}

void WhisperModel::execute(const Program& program, const Planned& planned, const Step& step,
                           cudaStream_t stream, const RunHooks* hooks) {
  using kernels::f16;
  using Clock = std::chrono::steady_clock;
  const int d = static_cast<int>(config_.d_model);
  const int n_ctx = static_cast<int>(config_.max_source_positions);
  const int encoder_heads = static_cast<int>(config_.encoder_heads);
  const int decoder_heads = static_cast<int>(config_.decoder_heads);
  const int decoder_head_dim = d / decoder_heads;
  const float eps = static_cast<float>(config_.layer_norm_eps);
  const bool timed = hooks != nullptr && hooks->times != nullptr;
  const auto at = [&](int id) { return value<f16>(planned, id); };
  const auto bias = [&](const Op& op) { return op.bias >= 0 ? weights_.data(op.bias) : nullptr; };
  const auto key_loads = [&](const KvView& view) {
    return limits_.wide_key_loads ? kernels::widest_key_loads(view, decoder_head_dim)
                                  : kernels::KeyLoads::Scalar;
  };

  for (const Op& op : program.ops) {
    const Value* out = op.out >= 0 ? &program.values[op.out] : nullptr;
    const Value* in0 = op.in0 >= 0 ? &program.values[op.in0] : nullptr;
    // Rows of the op: those of the value it writes, or of the value it reads when it writes none.
    const Value* shaped = out != nullptr ? out : in0;
    const int n_rows = static_cast<int>(row_count(shaped->rows, step.n_tokens, step.n_out));
    if (n_rows == 0) continue;

    const Clock::time_point start = timed ? Clock::now() : Clock::time_point{};
    switch (op.kind) {
      case OpKind::Embed:
        if (const WeightRef table = weights_.ref(op.weight); table.scheme == WeightScheme::F16) {
          kernels::embedding(weights_.data(op.weight), step.rows, at(op.out), n_rows, d, stream);
        } else {
          kernels::embedding_quantized(table, step.rows, at(op.out), n_rows, d, stream);
        }
        break;
      case OpKind::Linear:
        run_linear(weights_.ref(op.weight), bias(op), at(op.in0), value<void>(planned, op.out),
                   n_rows, static_cast<int>(in0->cols), static_cast<int>(out->cols), out->dtype,
                   limits_.linear, blas_, dequant_, stream);
        break;
      case OpKind::LayerNorm:
        kernels::layer_norm(at(op.in0), weights_.data(op.weight), bias(op), at(op.out), n_rows,
                            static_cast<int>(out->cols), eps, stream);
        break;
      case OpKind::AddLayerNorm:
        kernels::add_layer_norm(at(op.in0), at(op.in1), weights_.data(op.weight), bias(op),
                                at(op.out), n_rows, static_cast<int>(out->cols), eps, stream);
        break;
      case OpKind::Add:
        kernels::add(at(op.in0), at(op.in1), static_cast<std::size_t>(n_rows) * in0->cols, stream);
        break;
      case OpKind::Gelu:
        kernels::gelu(at(op.in0), at(op.out), static_cast<std::size_t>(n_rows) * in0->cols, stream);
        break;
      case OpKind::Unfold: {
        // Rows of one clip in the input and in the output; their ratio is the stride.
        const int in_frames = static_cast<int>(row_count(in0->rows, n_ctx, 0));
        const int out_frames = static_cast<int>(row_count(out->rows, n_ctx, 0));
        kernels::unfold3(at(op.in0), at(op.out), n_rows / out_frames, in_frames,
                         static_cast<int>(in0->cols), in_frames / out_frames, stream);
        break;
      }
      case OpKind::AddPositions:
        kernels::add_positions(at(op.in0), weights_.data(op.weight), step.rows, at(op.out), n_rows,
                               static_cast<int>(out->cols), stream);
        break;
      case OpKind::DenseAttention: {
        // The keys and values are read where the product left them, through a cache view laid
        // over q|k|v: one block of n_ctx positions per clip, block c for clip c.
        f16* qkv = at(op.in0);
        const KvView view{.k = qkv + d,
                          .v = qkv + 2 * d,
                          .block_table = clip_blocks_.as<std::int32_t>(),
                          .blocks_per_seq = 1,
                          .block_size = n_ctx,
                          .block_stride = static_cast<std::int64_t>(n_ctx) * 3 * d,
                          .head_stride = d / encoder_heads,
                          .pos_stride = 3 * d,
                          .dim_stride = 1};
        kernels::attention_dense(qkv, at(op.out), step.rows, step.tiles, step.n_tiles, view, 3 * d,
                                 n_ctx, encoder_heads, encoder_heads, d / encoder_heads, stream);
        break;
      }
      case OpKind::KvWrite:
        if (op.cache == KvCache::Cross) {
          // The halves of one stacked k|v value.
          kernels::kv_write(at(op.in0), at(op.in0) + d, step.rows, step.cross->layer_view(op.layer),
                            n_rows, decoder_heads, decoder_head_dim, 2 * d, stream);
        } else {
          kernels::kv_write(at(op.in0), at(op.in1), step.rows, step.self->layer_view(op.layer),
                            n_rows, decoder_heads, decoder_head_dim, d, stream);
        }
        break;
      case OpKind::Attention: {
        if (op.cache == KvCache::Cross) {
          // Every row reads all n_ctx positions of its sequence's block, one block per query.
          const KvView view = step.cross->layer_view(op.layer);
          kernels::attention(at(op.in0), at(op.out), step.rows, view, n_rows, decoder_heads,
                             decoder_heads, decoder_head_dim, n_ctx, n_ctx, key_loads(view),
                             stream);
          break;
        }
        const KvView view = step.self->layer_view(op.layer);
        if (step.n_tiles > 0 && kernels::attention_prefill_supports(view, decoder_head_dim)) {
          kernels::attention_prefill(at(op.in0), at(op.out), step.rows, step.tiles, step.n_tiles,
                                     view, decoder_heads, decoder_heads, decoder_head_dim, stream);
          break;
        }
        kernels::attention(at(op.in0), at(op.out), step.rows, view, n_rows, decoder_heads,
                           decoder_heads, decoder_head_dim, limits_.max_ctx, 0, key_loads(view),
                           stream);
        break;
      }
      case OpKind::GatherRows:
        kernels::gather_rows(at(op.in0), step.out_rows, at(op.out), n_rows,
                             static_cast<int>(out->cols), stream);
        break;
      case OpKind::SuppressTokens: {
        const Suppress& suppress = program.suppress;
        kernels::suppress_logits(
            value<float>(planned, op.in0), step.rows, step.out_rows, n_rows,
            static_cast<int>(in0->cols), suppress_always_.as<std::int32_t>(),
            static_cast<int>(suppress.always.size()), suppress_begin_.as<std::int32_t>(),
            static_cast<int>(suppress.begin.size()), suppress.begin_position, stream);
        break;
      }
      case OpKind::Argmax:
        kernels::argmax(value<float>(planned, op.in0), value<std::int32_t>(planned, op.out), n_rows,
                        static_cast<int>(in0->cols), stream);
        break;
      default:
        throw Error("a Whisper program has no " + std::string(op_name(op.kind)));
    }
    if (hooks == nullptr) continue;
    if (timed) {
      KVB_CUDA(cudaStreamSynchronize(stream));
      const std::size_t kind = static_cast<std::size_t>(op.kind);
      hooks->times->seconds[kind] += std::chrono::duration<double>(Clock::now() - start).count();
      hooks->times->calls[kind] += 1;
    }
    if (!hooks->on_site) continue;
    if (!op.in0_site.empty()) hooks->on_site(op.in0_site, op.in0, value<void>(planned, op.in0));
    if (!op.site.empty()) hooks->on_site(op.site, op.out, value<void>(planned, op.out));
  }
}

}  // namespace kvb::cuda
