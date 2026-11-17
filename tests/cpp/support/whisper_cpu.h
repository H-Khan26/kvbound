#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <numbers>
#include <span>
#include <string>
#include <vector>

#include "kvbound/dtype.h"
#include "kvbound/model_config.h"
#include "kvbound/program.h"
#include "kvbound/safetensors.h"
#include "kvbound/step.h"

// A CPU executor of Whisper programs: every op as kvbound/program.h defines it, in FP32 with
// FP64 sums, one row at a time. It exists to prove on the fixture that the programs compute the
// reference model; the engine's executor follows the same definitions on the GPU.
namespace kvb::test {

// Captured sites by name, each [rows, cols] row-major.
using Capture = std::map<std::string, std::vector<float>>;

// Both caches have one block per sequence and an identity block table: position p of
// block-table row s is slot s * max_target_positions + p of the Self cache, and encoder position
// p of that sequence's clip is slot s * max_source_positions + p of the Cross cache.
class CpuWhisper {
 public:
  CpuWhisper(const WhisperConfig& cfg, const WhisperPrograms& programs, const SafetensorsFile& file,
             int max_seqs)
      : cfg_(cfg), programs_(programs) {
    // One table serves both programs. The parts of a stacked weight sit end to end.
    for (const WeightParts& parts : programs.encoder.weights) {
      std::vector<float>& weight = weights_.emplace_back();
      for (const std::string& name : parts) {
        for (const std::uint16_t bits : file.at(name).as<std::uint16_t>()) {
          weight.push_back(half_bits_to_float(bits));
        }
      }
    }
    const std::size_t d = static_cast<std::size_t>(cfg.d_model);
    const std::size_t layers = static_cast<std::size_t>(cfg.decoder_layers);
    const std::size_t self_slots = static_cast<std::size_t>(max_seqs * cfg.max_target_positions);
    const std::size_t cross_slots = static_cast<std::size_t>(max_seqs * cfg.max_source_positions);
    self_k_.assign(layers, std::vector<float>(self_slots * d));
    self_v_ = self_k_;
    cross_k_.assign(layers, std::vector<float>(cross_slots * d));
    cross_v_ = cross_k_;
  }

  // Encodes rows.size() / max_source_positions clips. `mel` holds their features with one row
  // per frame, [clips * n_frames, n_mels].
  void encode(std::span<const StepRow> rows, std::span<const float> mel,
              Capture* capture = nullptr) {
    const Program& program = programs_.encoder;
    std::vector<std::vector<float>> values(program.values.size());
    values[program.input].assign(mel.begin(), mel.end());
    run(program, rows, {}, values, capture);
  }

  // Runs one decoder step and returns the next token of every output row.
  std::vector<std::int32_t> decode(std::span<const StepRow> rows,
                                   std::span<const std::int32_t> out_rows,
                                   Capture* capture = nullptr) {
    std::vector<std::vector<float>> values(programs_.decoder.values.size());
    return run(programs_.decoder, rows, out_rows, values, capture);
  }

 private:
  static double dot(const float* a, const float* b, std::int64_t n) {
    double sum = 0.0;
    for (std::int64_t i = 0; i < n; ++i) sum += static_cast<double>(a[i]) * b[i];
    return sum;
  }

  static void layer_norm(const float* x, const float* weight, const float* bias, double eps,
                         std::int64_t cols, float* out) {
    double mean = 0.0, variance = 0.0;
    for (std::int64_t c = 0; c < cols; ++c) mean += x[c];
    mean /= static_cast<double>(cols);
    for (std::int64_t c = 0; c < cols; ++c) variance += (x[c] - mean) * (x[c] - mean);
    const double scale = 1.0 / std::sqrt(variance / static_cast<double>(cols) + eps);
    for (std::int64_t c = 0; c < cols; ++c) {
      out[c] = static_cast<float>((x[c] - mean) * scale * weight[c] + bias[c]);
    }
  }

  // out = softmax(q . key(j) / sqrt(head_dim)) . value(j) over n_keys keys, head by head. Rows
  // are [n_heads, head_dim].
  template <typename Key, typename Val>
  void attend(const float* q, std::int64_t n_heads, std::int64_t n_keys, Key key, Val value,
              float* out) const {
    const std::int64_t head_dim = cfg_.d_model / n_heads;
    std::vector<double> prob(static_cast<std::size_t>(n_keys));
    for (std::int64_t h = 0; h < n_heads; ++h) {
      const std::int64_t at = h * head_dim;
      for (std::int64_t j = 0; j < n_keys; ++j) {
        prob[j] = dot(q + at, key(j) + at, head_dim) / std::sqrt(static_cast<double>(head_dim));
      }
      const double top = *std::max_element(prob.begin(), prob.end());
      double total = 0.0;
      for (double& p : prob) total += p = std::exp(p - top);
      for (std::int64_t i = 0; i < head_dim; ++i) {
        double sum = 0.0;
        for (std::int64_t j = 0; j < n_keys; ++j) sum += prob[j] * value(j)[at + i];
        out[at + i] = static_cast<float>(sum / total);
      }
    }
  }

  std::vector<std::int32_t> run(const Program& program, std::span<const StepRow> rows,
                                std::span<const std::int32_t> out_rows,
                                std::vector<std::vector<float>>& values, Capture* capture) {
    const std::int64_t n_tokens = static_cast<std::int64_t>(rows.size());
    const std::int64_t n_out = static_cast<std::int64_t>(out_rows.size());
    const std::int64_t d = cfg_.d_model;
    const std::int64_t n_ctx = cfg_.max_source_positions;
    const auto cols = [&](int id) { return program.values[id].cols; };
    const auto n_rows = [&](int id) { return row_count(program.values[id].rows, n_tokens, n_out); };
    const auto row = [&](int id, std::int64_t r) { return values[id].data() + r * cols(id); };
    std::vector<std::int32_t> next_ids;

    for (const Op& op : program.ops) {
      if (op.out >= 0 && op.out != op.in0) {
        values[op.out].assign(static_cast<std::size_t>(n_rows(op.out) * cols(op.out)), 0.0f);
      }
      const float* weight = op.weight >= 0 ? weights_[op.weight].data() : nullptr;
      const float* bias = op.bias >= 0 ? weights_[op.bias].data() : nullptr;
      switch (op.kind) {
        case OpKind::Embed:
          for (std::int64_t r = 0; r < n_tokens; ++r) {
            std::copy_n(weight + rows[r].token_id * d, d, row(op.out, r));
          }
          break;
        case OpKind::Linear:
          for (std::int64_t r = 0; r < n_rows(op.out); ++r) {
            for (std::int64_t o = 0; o < cols(op.out); ++o) {
              const double sum = dot(row(op.in0, r), weight + o * cols(op.in0), cols(op.in0));
              row(op.out, r)[o] = static_cast<float>(sum + (bias != nullptr ? bias[o] : 0.0f));
            }
          }
          break;
        case OpKind::Add:
        case OpKind::AddLayerNorm:
          for (std::size_t i = 0; i < values[op.in0].size(); ++i) {
            values[op.in0][i] += values[op.in1][i];
          }
          if (op.kind == OpKind::Add) break;
          [[fallthrough]];
        case OpKind::LayerNorm:
          for (std::int64_t r = 0; r < n_rows(op.out); ++r) {
            layer_norm(row(op.in0, r), weight, bias, cfg_.layer_norm_eps, cols(op.out),
                       row(op.out, r));
          }
          break;
        case OpKind::Gelu:
          for (float& x : values[op.in0]) {
            x = static_cast<float>(0.5 * x * (1.0 + std::erf(x / std::numbers::sqrt2)));
          }
          break;
        case OpKind::Unfold: {
          // Rows of one clip in the input and in the output; their ratio is the stride.
          const std::int64_t in_rows = row_count(program.values[op.in0].rows, n_ctx, 0);
          const std::int64_t clip_rows = row_count(program.values[op.out].rows, n_ctx, 0);
          const std::int64_t stride = in_rows / clip_rows;
          for (std::int64_t r = 0; r < n_rows(op.out); ++r) {
            const std::int64_t clip = r / clip_rows;
            for (std::int64_t k = 0; k < 3; ++k) {
              const std::int64_t source = (r % clip_rows) * stride + k - 1;
              if (source < 0 || source >= in_rows) continue;
              const float* in = row(op.in0, clip * in_rows + source);
              for (std::int64_t c = 0; c < cols(op.in0); ++c) row(op.out, r)[c * 3 + k] = in[c];
            }
          }
          break;
        }
        case OpKind::AddPositions:
          for (std::int64_t r = 0; r < n_tokens; ++r) {
            const float* position = weight + rows[r].position * d;
            for (std::int64_t c = 0; c < d; ++c) row(op.in0, r)[c] += position[c];
          }
          break;
        case OpKind::DenseAttention:
          for (std::int64_t r = 0; r < n_tokens; ++r) {
            const std::int64_t first = rows[r].seq * n_ctx;  // the clip's first row
            attend(
                row(op.in0, r), cfg_.encoder_heads, n_ctx,
                [&](std::int64_t j) { return row(op.in0, first + j) + d; },
                [&](std::int64_t j) { return row(op.in0, first + j) + 2 * d; }, row(op.out, r));
          }
          break;
        case OpKind::KvWrite:
          for (std::int64_t r = 0; r < n_tokens; ++r) {
            const std::int64_t at = rows[r].kv_slot * d;
            if (op.cache == KvCache::Cross) {
              std::copy_n(row(op.in0, r), d, cross_k_[op.layer].data() + at);
              std::copy_n(row(op.in0, r) + d, d, cross_v_[op.layer].data() + at);
            } else {
              std::copy_n(row(op.in0, r), d, self_k_[op.layer].data() + at);
              std::copy_n(row(op.in1, r), d, self_v_[op.layer].data() + at);
            }
          }
          break;
        case OpKind::Attention:
          for (std::int64_t r = 0; r < n_tokens; ++r) {
            const bool cross = op.cache == KvCache::Cross;
            const float* k = (cross ? cross_k_ : self_k_)[op.layer].data();
            const float* v = (cross ? cross_v_ : self_v_)[op.layer].data();
            // Cross attention reads the whole clip; self attention the row's own past.
            const std::int64_t first =
                rows[r].seq * (cross ? n_ctx : cfg_.max_target_positions) * d;
            attend(
                row(op.in0, r), cfg_.decoder_heads, cross ? n_ctx : rows[r].position + 1,
                [&](std::int64_t j) { return k + first + j * d; },
                [&](std::int64_t j) { return v + first + j * d; }, row(op.out, r));
          }
          break;
        case OpKind::GatherRows:
          for (std::int64_t o = 0; o < n_out; ++o) {
            std::copy_n(row(op.in0, out_rows[o]), d, row(op.out, o));
          }
          break;
        case OpKind::SuppressTokens:
          for (std::int64_t o = 0; o < n_out; ++o) {
            const float inf = std::numeric_limits<float>::infinity();
            for (const std::int32_t token : program.suppress.always) row(op.in0, o)[token] = -inf;
            if (rows[out_rows[o]].position != program.suppress.begin_position) continue;
            for (const std::int32_t token : program.suppress.begin) row(op.in0, o)[token] = -inf;
          }
          break;
        case OpKind::Argmax:
          for (std::int64_t o = 0; o < n_out; ++o) {
            const float* logits = row(op.in0, o);
            next_ids.push_back(static_cast<std::int32_t>(
                std::max_element(logits, logits + cols(op.in0)) - logits));
          }
          break;
        default:
          ADD_FAILURE() << "a Whisper program has no " << op_name(op.kind);
      }
      if (capture == nullptr) continue;
      if (!op.in0_site.empty()) (*capture)[op.in0_site] = values[op.in0];
      if (!op.site.empty()) (*capture)[op.site] = values[op.out];
    }
    return next_ids;
  }

  const WhisperConfig& cfg_;
  const WhisperPrograms& programs_;
  std::vector<std::vector<float>> weights_;
  // Per decoder layer, [slots, d_model].
  std::vector<std::vector<float>> self_k_, self_v_, cross_k_, cross_v_;
};

}  // namespace kvb::test
