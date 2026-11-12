#include "kvbound/cuda/llama_model.h"

#include <chrono>
#include <cmath>
#include <numbers>

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

}  // namespace

LlamaModel::LlamaModel(const std::filesystem::path& model_dir, const LlamaModelConfig& limits,
                       cudaStream_t stream)
    : config_(load_model_config(model_dir / "config.json")),
      limits_(limits),
      program_(build_llama_program(config_)),
      value_bytes_(value_bytes(program_, limits.max_step_tokens, limits.max_outputs)),
      plan_(plan_memory(value_bytes_, value_lifetimes(program_), limits.plan)),
      weights_(SafetensorsFile(model_dir / "model.safetensors"), program_.weights, stream),
      arena_(plan_.arena_bytes),
      blas_(stream, limits.gemm) {
  const std::vector<float> table = rope_cos_sin(config_, limits.max_ctx);
  rope_table_ = DeviceBuffer(table.size() * sizeof(float));
  KVB_CUDA(cudaMemcpy(rope_table_.as(), table.data(), rope_table_.bytes(), cudaMemcpyHostToDevice));
}

void LlamaModel::run(const StepRow* rows, const std::int32_t* out_rows, int n_tokens, int n_out,
                     const KvPool& kv, cudaStream_t stream, const RunHooks* hooks) {
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
        kernels::embedding(weights_.data(op.weight), rows, value<f16>(op.out), n_tokens,
                           static_cast<int>(out->cols), stream);
        break;
      case OpKind::RmsNorm:
        kernels::rms_norm(value<f16>(op.in0), weights_.data(op.weight), value<f16>(op.out), n_rows,
                          static_cast<int>(out->cols), static_cast<float>(config_.rms_norm_eps),
                          stream);
        break;
      case OpKind::Linear:
        blas_.linear(weights_.data(op.weight), value<f16>(op.in0), value<void>(op.out), n_rows,
                     static_cast<int>(in0->cols), static_cast<int>(out->cols), out->dtype);
        break;
      case OpKind::Rope:
        kernels::rope(value<f16>(op.in0), rows, rope_table_.as<float>(), n_tokens,
                      static_cast<int>(in0->cols) / head_dim, head_dim, stream);
        break;
      case OpKind::KvWrite:
        kernels::kv_write(value<f16>(op.in0), value<f16>(op.in1), rows, kv.layer_view(op.layer),
                          n_tokens, n_kv_heads, head_dim, stream);
        break;
      case OpKind::Attention:
        kernels::attention(value<f16>(op.in0), value<f16>(op.out), rows, kv.layer_view(op.layer),
                           n_tokens, n_heads, n_kv_heads, head_dim, limits_.max_ctx, stream);
        break;
      case OpKind::SiluMul:
        kernels::silu_mul(value<f16>(op.in0), value<f16>(op.in1),
                          static_cast<std::size_t>(n_rows) * in0->cols, stream);
        break;
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
    if (!op.site.empty() && hooks->on_site) hooks->on_site(op, value<void>(op.out));
  }
}

}  // namespace kvb::cuda
