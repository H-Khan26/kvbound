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
  Linear,       // out = in0 * weight^T, plus bias when the op has one
  Rope,         // rotates in0 in place by each row's position
  KvWrite,      // stores keys in0 and values in1 into the KV cache `cache`
  Attention,    // out = softmax(in0 . cached keys) * cached values, from the KV cache `cache`
  SiluMul,      // out = silu(gate) * up: gate in0 and up in1, or both halves of in0 when in1 < 0
  Add,          // in0 += in1, in place
  GatherRows,   // out = the rows of in0 whose logits are requested
  Argmax,       // out = index of the largest element of each row of in0
  AddRmsNorm,   // in0 += in1 in place, then out = RmsNorm(in0)
  RopeKvWrite,  // in0 holds q|k|v per row: out = rotated q; rotated k and v go to the KV cache
  // The kinds below appear in Whisper programs only; "Whisper ops" further down gives their
  // fields.
  LayerNorm,       // out = (in0 - mean) * rsqrt(variance + eps) * weight + bias
  AddLayerNorm,    // in0 += in1 in place, then out = LayerNorm(in0)
  Gelu,            // in0 = gelu(in0) in place, the exact erf form
  Unfold,          // out = each row of in0 with its two neighbours: the input of a 3-tap conv
  AddPositions,    // in0 += weight[each row's position], in place
  DenseAttention,  // in0 holds q|k|v per row: out = attention over every row of the sequence
  SuppressTokens,  // in0 = -inf where a token may not be generated, in place
};
inline constexpr int kOpKindCount = 19;

std::string_view op_name(OpKind kind) noexcept;

// How many rows a value has in a step: one per token, one per requested output, or one per mel
// frame of a speech encoder step.
enum class RowKind : std::uint8_t { Token, Output, Frame };

// Frame rows per token row. A Whisper encoder position covers two mel frames, because the second
// convolution of its stem has stride 2.
inline constexpr int kFramesPerToken = 2;

// Rows of a value of this kind in a step of n_tokens rows with n_outputs requested outputs.
constexpr std::int64_t row_count(RowKind kind, std::int64_t n_tokens,
                                 std::int64_t n_outputs) noexcept {
  switch (kind) {
    case RowKind::Output:
      return n_outputs;
    case RowKind::Frame:
      return kFramesPerToken * n_tokens;
    case RowKind::Token:
      break;
  }
  return n_tokens;
}

struct Value {
  std::string name;
  RowKind rows{};
  std::int64_t cols = 0;
  DType dtype{};
};

// Which cache a KvWrite or an Attention uses. Self holds the keys and values of the tokens a
// sequence has processed. Cross holds, per sequence, the keys and values of every encoder
// position of its audio; a decoder reads all of them, whatever the row's position.
enum class KvCache : std::uint8_t { Self, Cross };

struct Op {
  OpKind kind{};
  int layer = -1;  // transformer layer, -1 outside the layers
  int in0 = -1;    // value ids read
  int in1 = -1;
  int out = -1;     // value id written; equals in0 for in-place ops, -1 for KvWrite
  int weight = -1;  // index into Program::weights
  int bias = -1;    // index into Program::weights of a one-dimensional tensor, -1 for none
  KvCache cache = KvCache::Self;
  std::string site;      // golden capture-site name of `out` after this op, empty if none
  std::string in0_site;  // capture-site name of in0 after an op that updates it in place
};

// A tensor the program reads: one checkpoint tensor, or several with the same row length stacked
// along their rows, so that one product computes all of them. The parts of a stacked bias are
// the biases of the stacked matrices, end to end.
using WeightParts = std::vector<std::string>;

// The tokens SuppressTokens masks in each output row.
struct Suppress {
  std::vector<std::int32_t> always;  // in every row
  std::vector<std::int32_t> begin;   // only in a row at begin_position
  // Position of the last prompt token, whose row produces the first generated token.
  std::int32_t begin_position = -1;
};

struct Program {
  std::vector<Value> values;
  std::vector<Op> ops;
  std::vector<WeightParts> weights;  // checkpoint tensor names
  // Per weight, the rows of each stacked part in order (for a bias, its elements); empty for a
  // single tensor.
  std::vector<std::vector<std::int64_t>> part_rows;
  int input = -1;   // value id the host fills before the first op, -1 if the program has none
  int logits = -1;  // value ids of the outputs, -1 if the program has none
  int next_ids = -1;
  Suppress suppress;
};

// With `fuse`, each layer stacks its q, k and v projections and its gate and up projections into
// one product each, folds each residual add into the norm that follows it, and rotates q and k
// as it writes the cache: 9 ops per layer instead of 16. The capture sites are the same.
Program build_llama_program(const ModelConfig& cfg, bool fuse = true);

// A Whisper model is two programs. The encoder runs once per audio clip and leaves the keys and
// values of every decoder layer's cross attention in the Cross cache; the decoder then runs
// step after step like a Llama program. Their value ids are separate, so each gets its own
// memory plan. Their weight tables are the same table, so an op's `weight` and `bias` index one
// set of device weights, loaded once.
//
// The Cross cache gives every decoder sequence one block of n_ctx slots, where n_ctx is
// max_source_positions: encoder position p of the sequence's clip lives in slot
// block * n_ctx + p. A decoder row finds the block through its StepRow.seq, the same block-table
// row its Self cache uses. An encoder row names the slot itself, because its seq means
// something else.
//
// Encoder steps. A step encodes n_audio clips and has n_audio * n_ctx token rows. The rows of a
// clip are consecutive and in position order:
//   StepRow.position  0 .. n_ctx - 1
//   StepRow.seq       index of the clip within the step, 0 .. n_audio - 1; not a block-table row
//   StepRow.kv_slot   block * n_ctx + position, for the Cross block of the sequence that owns
//                     the clip
//   StepRow.token_id  unused
// Program::input is the log-mel features, a Frame value [n_audio * 2 * n_ctx, n_mels] in FP16:
// row clip * 2 * n_ctx + frame holds every mel bin of one frame, so the host transposes the
// usual [n_mels, frames] layout. Its capture sites are "encoder.*"; all have one row per
// position except "encoder.conv1", which has one per frame.
//
// Decoder steps are Llama steps: rows of any number of sequences, each with a token id, a
// position, a block-table row and a Self cache slot. Its capture sites are "decoder.*" and
// "logits".
//
// Whisper ops. Where a field is not named it is unused.
//   Linear          `bias` >= 0 adds weights[bias] to every row of the product. The stem's conv
//                   weights are stored [d_model, channels, 3] and read as matrices
//                   [d_model, channels * 3], which is the column order Unfold writes.
//   LayerNorm       out = norm of in0, with scale weights[weight] and shift weights[bias]; the
//                   mean and variance of a row are taken over its columns, eps is layer_norm_eps.
//   AddLayerNorm    in0 += in1 in place, then LayerNorm as above. `site` names out and
//                   `in0_site` names in0 after the add.
//   Gelu            in0 = out, in place.
//   Unfold          out[r][c * 3 + k] = in0[r * stride + k - 1][c] for k in 0..2, and 0 where
//                   that row lies outside r's clip. out has 3 * in0.cols columns. The stride is
//                   1 when in0 and out are both Frame values and kFramesPerToken from a Frame
//                   value to a Token value.
//   AddPositions    in0 = out, in place: row r gains row StepRow.position of weights[weight], a
//                   stored table [positions, d_model].
//   DenseAttention  Encoder self attention over encoder_heads heads. in0 has 3 * d_model
//                   columns, q|k|v of each row; out has d_model. A row attends to all n_ctx rows
//                   of its clip, with no mask and no cache: keys and values are read from in0.
//   KvWrite         Decoder self attention: keys in0 and values in1 go to the Self cache at
//                   StepRow.kv_slot, as in Llama. With cache = Cross (encoder program): in1 is
//                   -1 and in0 has 2 * d_model columns, k|v of each row, which go to the Cross
//                   cache of decoder layer `layer` at StepRow.kv_slot.
//   Attention       cache = Self is Llama's attention: a row reads the cached positions of its
//                   sequence up to its own. cache = Cross: a row reads all n_ctx positions of
//                   the Cross cache block of StepRow.seq, in decoder layer `layer`, with no mask.
//                   Either way in0 is the queries and the heads are decoder_heads.
//   SuppressTokens  in0 = out = the FP32 logits, in place: in output row o, the tokens of
//                   Program::suppress.always become -inf, and those of suppress.begin too when
//                   the row's StepRow.position is suppress.begin_position. The "logits" site is
//                   captured before this op; Program::logits read after the step is masked.
struct WhisperPrograms {
  Program encoder;
  Program decoder;
};

// Throws kvb::Error if the rules do not fit the model. With `fuse`, each residual add is folded
// into the layer norm that follows it (AddLayerNorm), except the decoder's last, whose norm runs
// on the gathered output rows only. The capture sites and the weight table are the same.
WhisperPrograms build_whisper_programs(const WhisperConfig& cfg, const WhisperRules& rules,
                                       bool fuse = true);

}  // namespace kvb
