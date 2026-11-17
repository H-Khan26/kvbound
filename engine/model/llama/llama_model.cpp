#include "kvbound/cuda/llama_model.h"

#include <chrono>
#include <cmath>
#include <numbers>
#include <string>

#include "kvbound/cuda/kernels.h"

namespace kvb::cuda {
namespace {

// {cos, sin} of position * inv_freq_j for every position and rotation pair, in FP32 like the
// reference model: inv_freq_j = theta^(-2j / head_dim), optionally rescaled the Llama 3 way.
std::vector<float> rope_cos_sin(const ModelConfig& cfg, int max_positions) {
  const int head_dim = static_cast<int>(cfg.head_dim());
  const int half_dim = head_dim / 2;
  std::vector<float> inv_freq(half_dim);
  for (int j = 0; j < half_dim; ++j) {
    const float exponent = static_cast<float>(2 * j) / static_cast<float>(head_dim);
    inv_freq[j] = 1.0f / std::pow(static_cast<float>(cfg.rope_theta), exponent);
  }
  if (cfg.rope_scaling) {
    const Llama3RopeScaling& s = *cfg.rope_scaling;
    const float original = static_cast<float>(s.original_max_position_embeddings);
    const float low = static_cast<float>(s.low_freq_factor);
    const float high = static_cast<float>(s.high_freq_factor);
    const float factor = static_cast<float>(s.factor);
    for (float& freq : inv_freq) {
      const float wavelen = 2.0f * std::numbers::pi_v<float> / freq;
      if (wavelen > original / low) {
        freq /= factor;
      } else if (wavelen >= original / high) {
        const float smooth = (original / wavelen - low) / (high - low);
        freq = (1.0f - smooth) * freq / factor + smooth * freq;
      }
    }
  }

  std::vector<float> table(static_cast<std::size_t>(max_positions) * half_dim * 2);
  for (int pos = 0; pos < max_positions; ++pos) {
    for (int j = 0; j < half_dim; ++j) {
      const float angle = static_cast<float>(pos) * inv_freq[j];
      const std::size_t at = (static_cast<std::size_t>(pos) * half_dim + j) * 2;
      table[at] = std::cos(angle);
      table[at + 1] = std::sin(angle);
    }
  }
  return table;
}

LlamaModelConfig checked(const LlamaModelConfig& limits, const ModelConfig& cfg) {
  const int max_positions = kernels::attention_max_positions(static_cast<int>(cfg.head_dim()));
  if (limits.max_ctx <= 0 || limits.max_ctx > max_positions) {
    throw Error("max_ctx must be between 1 and " + std::to_string(max_positions) +
                " for head_dim " + std::to_string(cfg.head_dim()));
  }
  return limits;
}

// Opens the checkpoint and checks every tensor the program reads against the shape the program
// will use it with, so a config that does not match its weights fails here and not as an
// out-of-bounds read on the device.
DeviceWeights load_weights(const std::filesystem::path& model_dir, const Program& program,
                           const ModelConfig& cfg, cudaStream_t stream) {
  const SafetensorsFile file(model_dir / "model.safetensors");
  const WeightFormat format = weight_format_from_metadata(file.metadata());
  for (const Op& op : program.ops) {
    if (op.weight < 0) continue;
    const std::int64_t out_cols = program.values[op.out].cols;
    const WeightParts& parts = program.weights[op.weight];
    const std::vector<std::int64_t>& part_rows = program.part_rows[op.weight];
    if (parts.size() > 1 && (op.kind != OpKind::Linear || part_rows.size() != parts.size())) {
      throw Error(parts[0] + ": only matrix products read stacked weights, with rows per part");
    }
    // Rows of the stacked matrix, summed over its parts.
    std::int64_t stacked_rows = 0;
    for (std::size_t part = 0; part < parts.size(); ++part) {
      const std::string& name = parts[part];
      const TensorView& tensor = file.at(name);
      const bool integers = tensor.dtype == DType::I8 || tensor.dtype == DType::U8;
      if (op.kind == OpKind::RmsNorm || op.kind == OpKind::AddRmsNorm) {
        // Only matrices are quantized: the norm kernel reads its weight as FP16.
        if (integers) throw Error(name + ": a one-dimensional weight must be floating point");
        if (tensor.shape != std::vector<std::int64_t>{out_cols}) {
          throw Error(name + ": shape does not match config.json");
        }
        continue;
      }
      // A matrix [rows, cols]: the embedding table, or a linear layer's weight.
      const std::int64_t cols = op.kind == OpKind::Embed ? out_cols : program.values[op.in0].cols;
      const std::int64_t rows = op.kind == OpKind::Embed ? cfg.vocab_size
                                : parts.size() == 1      ? out_cols
                                                         : part_rows[part];
      std::vector<std::int64_t> expected{rows, cols};
      if (integers) {
        expected = stored_shape(format, rows, cols);
        if (file.at(name + ".scale").shape != scale_shape(format, rows, cols)) {
          throw Error(name + ".scale: shape does not match the weight format");
        }
      }
      if (tensor.shape != expected) throw Error(name + ": shape does not match config.json");
      stacked_rows += rows;
    }
    if (op.kind == OpKind::Linear && stacked_rows != out_cols) {
      throw Error(parts[0] + ": stacked rows do not match config.json");
    }
  }
  return DeviceWeights(file, program.weights, stream);
}

// Bytes of scratch for expanding quantized matrices: the largest one a step may multiply with
// more rows than the in-place kernel takes, up to one slice. None unless the engine expands.
std::size_t dequant_scratch_bytes(const Program& program, const DeviceWeights& weights,
                                  const LlamaModelConfig& limits) {
  if (!limits.expand_quantized) return 0;
  std::size_t bytes = 0;
  for (const Op& op : program.ops) {
    if (op.kind != OpKind::Linear || weights.ref(op.weight).scheme == WeightScheme::F16) continue;
    const Value& out = program.values[op.out];
    const int max_rows = out.rows == RowKind::Token ? limits.max_step_tokens : limits.max_outputs;
    if (max_rows <= limits.quantized_gemm_rows) continue;
    const std::size_t row_bytes = program.values[op.in0].cols * sizeof(std::uint16_t);
    const std::size_t matrix = static_cast<std::size_t>(out.cols) * row_bytes;
    bytes = std::max(bytes, std::min(matrix, std::max(limits.dequant_slice_bytes, row_bytes)));
  }
  return bytes;
}

}  // namespace

LlamaModel::LlamaModel(const std::filesystem::path& model_dir, const LlamaModelConfig& limits,
                       cudaStream_t stream)
    : config_(load_model_config(model_dir / "config.json")),
      limits_(checked(limits, config_)),
      program_(build_llama_program(config_, limits.fuse)),
      value_bytes_(value_bytes(program_, limits.max_step_tokens, limits.max_outputs)),
      plan_(plan_memory(value_bytes_, value_lifetimes(program_), limits.plan)),
      weights_(load_weights(model_dir, program_, config_, stream)),
      arena_(plan_.arena_bytes),
      dequant_(dequant_scratch_bytes(program_, weights_, limits_)),
      blas_(stream, limits.gemm) {
  const std::vector<float> table = rope_cos_sin(config_, limits.max_ctx);
  rope_table_ = DeviceBuffer(table.size() * sizeof(float));
  KVB_CUDA(cudaMemcpy(rope_table_.as(), table.data(), rope_table_.bytes(), cudaMemcpyHostToDevice));
}

void LlamaModel::run(const StepRow* rows, const std::int32_t* out_rows, int n_tokens, int n_out,
                     const AttnTile* tiles, int n_tiles, const KvPool& kv, cudaStream_t stream,
                     const RunHooks* hooks) {
  using kernels::f16;
  using Clock = std::chrono::steady_clock;
  const int n_heads = static_cast<int>(config_.n_heads);
  const int n_kv_heads = static_cast<int>(config_.n_kv_heads);
  const int head_dim = static_cast<int>(config_.head_dim());
  const bool timed = hooks != nullptr && hooks->times != nullptr;

  for (const Op& op : program_.ops) {
    const Value* out = op.out >= 0 ? &program_.values[op.out] : nullptr;
    const Value* in0 = op.in0 >= 0 ? &program_.values[op.in0] : nullptr;
    // Row count of the op: outputs are produced for n_out rows, everything else per token.
    const Value* shaped = out != nullptr ? out : in0;
    const int n_rows = shaped->rows == RowKind::Token ? n_tokens : n_out;
    if (n_rows == 0) continue;

    const Clock::time_point start = timed ? Clock::now() : Clock::time_point{};
    switch (op.kind) {
      case OpKind::Embed:
        if (const WeightRef table = weights_.ref(op.weight); table.scheme == WeightScheme::F16) {
          kernels::embedding(weights_.data(op.weight), rows, value<f16>(op.out), n_tokens,
                             static_cast<int>(out->cols), stream);
        } else {
          kernels::embedding_quantized(table, rows, value<f16>(op.out), n_tokens,
                                       static_cast<int>(out->cols), stream);
        }
        break;
      case OpKind::RmsNorm:
        kernels::rms_norm(value<f16>(op.in0), weights_.data(op.weight), value<f16>(op.out), n_rows,
                          static_cast<int>(out->cols), static_cast<float>(config_.rms_norm_eps),
                          stream);
        break;
      case OpKind::AddRmsNorm:
        kernels::add_rms_norm(value<f16>(op.in0), value<f16>(op.in1), weights_.data(op.weight),
                              value<f16>(op.out), n_rows, static_cast<int>(out->cols),
                              static_cast<float>(config_.rms_norm_eps), stream);
        break;
      case OpKind::Linear: {
        const WeightRef weight = weights_.ref(op.weight);
        const int n_in = static_cast<int>(in0->cols), n_out_cols = static_cast<int>(out->cols);
        if (weight.scheme == WeightScheme::F16 && n_rows <= limits_.f16_gemv_rows &&
            n_in % 8 == 0) {
          kernels::linear_f16(weights_.data(op.weight), value<f16>(op.in0), value<void>(op.out),
                              n_rows, n_in, n_out_cols, out->dtype == DType::F32, stream);
        } else if (weight.scheme == WeightScheme::F16) {
          blas_.linear(weights_.data(op.weight), value<f16>(op.in0), value<void>(op.out), n_rows,
                       n_in, n_out_cols, out->dtype);
        } else if (n_rows <= limits_.quantized_gemm_rows) {
          kernels::linear_quantized(weight, value<f16>(op.in0), value<void>(op.out), n_rows, n_in,
                                    n_out_cols, out->dtype == DType::F32, stream);
        } else if (!limits_.expand_quantized) {
          kernels::gemm_quantized(weight, value<f16>(op.in0), value<void>(op.out), n_rows, n_in,
                                  n_out_cols, out->dtype == DType::F32, stream);
        } else {
          // Expanded and multiplied a slice of output channels at a time.
          const int slice = static_cast<int>(dequant_.bytes() / sizeof(f16) / n_in);
          const std::size_t element = size_bytes(out->dtype);
          for (int first = 0; first < n_out_cols; first += slice) {
            const int count = std::min(slice, n_out_cols - first);
            kernels::dequantize(weight, dequant_.as<f16>(), n_in, first, count, stream);
            blas_.linear(dequant_.as<f16>(), value<f16>(op.in0),
                         value<std::byte>(op.out) + first * element, n_rows, n_in, count,
                         out->dtype, n_out_cols);
          }
        }
        break;
      }
      case OpKind::Rope:
        kernels::rope(value<f16>(op.in0), rows, rope_table_.as<float>(), n_tokens,
                      static_cast<int>(in0->cols) / head_dim, head_dim, stream);
        break;
      case OpKind::KvWrite:
        kernels::kv_write(value<f16>(op.in0), value<f16>(op.in1), rows, kv.layer_view(op.layer),
                          n_tokens, n_kv_heads, head_dim, stream);
        break;
      case OpKind::RopeKvWrite:
        kernels::rope_kv_write(value<f16>(op.in0), rows, rope_table_.as<float>(),
                               value<f16>(op.out), kv.layer_view(op.layer), n_tokens, n_heads,
                               n_kv_heads, head_dim, stream);
        break;
      case OpKind::Attention: {
        const KvView view = kv.layer_view(op.layer);
        if (n_tiles > 0 && kernels::attention_prefill_supports(view, head_dim)) {
          kernels::attention_prefill(value<f16>(op.in0), value<f16>(op.out), rows, tiles, n_tiles,
                                     view, n_heads, n_kv_heads, head_dim, stream);
          break;
        }
        kernels::attention(value<f16>(op.in0), value<f16>(op.out), rows, view, n_tokens, n_heads,
                           n_kv_heads, head_dim, limits_.max_ctx,
                           limits_.wide_key_loads ? kernels::widest_key_loads(view, head_dim)
                                                  : kernels::KeyLoads::Scalar,
                           stream);
        break;
      }
      case OpKind::SiluMul: {
        // Separate gate and up values, or both halves of each row of the stacked product.
        const int dim = static_cast<int>(out->cols);
        const f16* gate = value<f16>(op.in0);
        const f16* up = op.in1 >= 0 ? value<f16>(op.in1) : gate + dim;
        kernels::silu_mul(gate, up, value<f16>(op.out), n_rows, dim, static_cast<int>(in0->cols),
                          stream);
        break;
      }
      case OpKind::Add:
        kernels::add(value<f16>(op.in0), value<f16>(op.in1),
                     static_cast<std::size_t>(n_rows) * in0->cols, stream);
        break;
      case OpKind::GatherRows:
        kernels::gather_rows(value<f16>(op.in0), out_rows, value<f16>(op.out), n_out,
                             static_cast<int>(out->cols), stream);
        break;
      case OpKind::Argmax:
        kernels::argmax(value<float>(op.in0), value<std::int32_t>(op.out), n_out,
                        static_cast<int>(in0->cols), stream);
        break;
    }
    if (hooks == nullptr) continue;
    if (timed) {
      KVB_CUDA(cudaStreamSynchronize(stream));
      const std::size_t kind = static_cast<std::size_t>(op.kind);
      hooks->times->seconds[kind] += std::chrono::duration<double>(Clock::now() - start).count();
      hooks->times->calls[kind] += 1;
    }
    if (!hooks->on_site) continue;
    if (!op.in0_site.empty()) hooks->on_site(op.in0_site, op.in0, value<void>(op.in0));
    if (!op.site.empty()) hooks->on_site(op.site, op.out, value<void>(op.out));
  }
}

}  // namespace kvb::cuda
