#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "kvbound/dtype.h"
#include "kvbound/model_config.h"

namespace kvb {

// One forward pass written out as a flat list of operations over named intermediate values.
// The list is plain data built once per model: the executor walks it to launch kernels, the
// memory planner reads value lifetimes from it, and tests use its capture-site names.

enum class OpKind : std::uint8_t {
  Embed,        // out = embedding[token ids]
  RmsNorm,      // out = in0 * rsqrt(mean(in0^2) + eps) * weight
  Linear,       // out = in0 * weight^T
  Rope,         // rotates in0 in place by each row's position
  KvWrite,      // stores keys in0 and values in1 into the KV cache
  Attention,    // out = softmax(in0 . cached keys) * cached values
  SiluMul,      // out = silu(gate) * up: gate in0 and up in1, or both halves of in0 when in1 < 0
  Add,          // in0 += in1, in place
  GatherRows,   // out = the rows of in0 whose logits are requested
  Argmax,       // out = index of the largest element of each row of in0
  AddRmsNorm,   // in0 += in1 in place, then out = RmsNorm(in0)
  RopeKvWrite,  // in0 holds q|k|v per row: out = rotated q; rotated k and v go to the KV cache
};
inline constexpr int kOpKindCount = 12;

std::string_view op_name(OpKind kind) noexcept;

// How many rows a value has in a step: one per token, or one per requested output.
enum class RowKind : std::uint8_t { Token, Output };

struct Value {
  std::string name;
  RowKind rows{};
  std::int64_t cols = 0;
  DType dtype{};
};

struct Op {
  OpKind kind{};
  int layer = -1;  // transformer layer, -1 outside the layers
  int in0 = -1;    // value ids read
  int in1 = -1;
  int out = -1;          // value id written; equals in0 for in-place ops, -1 for KvWrite
  int weight = -1;       // index into Program::weights
  std::string site;      // golden capture-site name of `out` after this op, empty if none
  std::string in0_site;  // capture-site name of in0 after an op that updates it in place
};

// A tensor the program reads: one checkpoint tensor, or several matrices with the same row length
// stacked along their rows, so that one product computes all of them.
using WeightParts = std::vector<std::string>;

struct Program {
  std::vector<Value> values;
  std::vector<Op> ops;
  std::vector<WeightParts> weights;  // checkpoint tensor names
  int logits = -1;                   // value ids of the outputs
  int next_ids = -1;
};

// With `fuse`, each layer stacks its q, k and v projections and its gate and up projections into
// one product each, folds each residual add into the norm that follows it, and rotates q and k
// as it writes the cache: 9 ops per layer instead of 16. The capture sites are the same.
Program build_llama_program(const ModelConfig& cfg, bool fuse = true);

}  // namespace kvb
