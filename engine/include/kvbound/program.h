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
  Embed,       // out = embedding[token ids]
  RmsNorm,     // out = in0 * rsqrt(mean(in0^2) + eps) * weight
  Linear,      // out = in0 * weight^T
  Rope,        // rotates in0 in place by each row's position
  KvWrite,     // stores keys in0 and values in1 into the KV cache
  Attention,   // out = softmax(in0 . cached keys) * cached values
  SiluMul,     // in0 = silu(in0) * in1, in place
  Add,         // in0 += in1, in place
  GatherRows,  // out = the rows of in0 whose logits are requested
  Argmax,      // out = index of the largest element of each row of in0
};
inline constexpr int kOpKindCount = 10;

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
  int out = -1;      // value id written; equals in0 for in-place ops, -1 for KvWrite
  int weight = -1;   // index into Program::weights
  std::string site;  // golden capture-site name of `out` after this op, empty if none
};

struct Program {
  std::vector<Value> values;
  std::vector<Op> ops;
  std::vector<std::string> weights;  // checkpoint tensor names
  int logits = -1;                   // value ids of the outputs
  int next_ids = -1;
};

Program build_llama_program(const ModelConfig& cfg);

}  // namespace kvb
