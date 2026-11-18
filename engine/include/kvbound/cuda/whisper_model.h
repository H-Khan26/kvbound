#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <filesystem>
#include <vector>

#include "kvbound/cuda/blas.h"
#include "kvbound/cuda/device.h"
#include "kvbound/cuda/kv_pool.h"
#include "kvbound/cuda/linear.h"
#include "kvbound/cuda/run_hooks.h"
#include "kvbound/cuda/weights.h"
#include "kvbound/memory_plan.h"
#include "kvbound/model_config.h"
#include "kvbound/program.h"
#include "kvbound/step.h"

namespace kvb::cuda {

struct WhisperModelConfig {
  int max_ctx = 0;           // positions a decoder sequence may reach
  int max_step_tokens = 0;   // rows of the largest decoder step
  int max_outputs = 0;       // logit rows of the largest decoder step
  int max_encode_batch = 1;  // clips of the largest encoder step
  PlanStrategy plan = PlanStrategy::GreedyBySize;
  GemmAccumulate gemm = GemmAccumulate::F32;
  // Build the fused programs: residual adds folded into the layer norms that follow them.
  bool fuse = true;
  LinearOptions linear;  // which kernel multiplies by each matrix
  // Read cached keys 16 bytes at a time where the KV layout allows it.
  bool wide_key_loads = true;
  // An expanded matrix is written this many bytes at a time, as whole output channels.
  std::size_t dequant_slice_bytes = std::size_t{32} << 20;
};

// Opens <model_dir>/model.safetensors and checks every tensor the two programs read against the
// shape the programs use it with, so a config that does not match its weights fails here and
// not as an out-of-bounds read on the device. A matrix may be quantized; a convolution weight
// [out, in, 3], a position table, a bias and a layer-norm parameter must be floating point.
DeviceWeights load_whisper_weights(const std::filesystem::path& model_dir,
                                   const WhisperPrograms& programs, const WhisperConfig& cfg,
                                   cudaStream_t stream);

// A Whisper checkpoint on the device plus everything its two passes need: the encoder and
// decoder programs, one activation arena that both use in turn, and the decoding rules.
// Stateless between steps; the caches and each step's rows are passed in.
class WhisperModel {
 public:
  WhisperModel(const std::filesystem::path& model_dir, const WhisperModelConfig& config,
               cudaStream_t stream);

  const WhisperConfig& config() const noexcept { return config_; }
  const WhisperRules& rules() const noexcept { return rules_; }
  const WhisperPrograms& programs() const noexcept { return programs_; }
  // Where each value of a program lives in the arena, and its size at the largest step.
  const MemoryPlan& encoder_plan() const noexcept { return encoder_.plan; }
  const MemoryPlan& decoder_plan() const noexcept { return decoder_.plan; }
  // The arena holds the larger of the two plans: the passes never overlap, and what the encoder
  // leaves for the decoder is in the Cross cache.
  std::size_t arena_bytes() const noexcept { return arena_.bytes(); }
  std::size_t weight_bytes() const noexcept { return weights_.bytes(); }
  const WeightFormat& weight_format() const noexcept { return weights_.format(); }
  // Scratch that holds one quantized matrix expanded to FP16; empty when no step needs it.
  std::size_t dequant_bytes() const noexcept { return dequant_.bytes(); }

  // The encoder's input on the device: log-mel features [n_audio * n_frames, n_mels] in FP16,
  // one row per frame. The caller fills it before encode().
  std::uint16_t* mel() const noexcept {
    return value<std::uint16_t>(encoder_, programs_.encoder.input);
  }

  // Launches the encoder program over n_audio clips. rows [n_audio * max_source_positions] and
  // tiles [n_tiles] are device arrays: the rows as program.h describes an encoder step, the
  // tiles from plan_attention_tiles(rows, kAttnTileRows). Fills the Cross cache blocks the rows
  // name. Returns without waiting unless hooks ask for timing or captures.
  void encode(const StepRow* rows, int n_audio, const AttnTile* tiles, int n_tiles,
              const KvPool& cross, cudaStream_t stream, const RunHooks* hooks = nullptr);

  // Launches the decoder program for one step, as LlamaModel::run does: rows [n_tokens] and
  // out_rows [n_out] are device arrays, and with n_tiles > 0 self attention runs over tiles.
  // Every row's sequence must have been encoded into the Cross cache block of its StepRow.seq.
  // Without `suppress` the SuppressTokens op is skipped: the logits stay as computed and the
  // argmax reads them unmasked.
  void run(const StepRow* rows, const std::int32_t* out_rows, int n_tokens, int n_out,
           const AttnTile* tiles, int n_tiles, const KvPool& self, const KvPool& cross,
           cudaStream_t stream, const RunHooks* hooks = nullptr, bool suppress = true);

  // Device outputs of the last run(): logits [n_out, vocab] with suppressed tokens at -inf unless
  // it ran without `suppress`, and their argmax [n_out]. A later encode() overwrites them.
  const float* logits() const noexcept { return value<float>(decoder_, programs_.decoder.logits); }
  const std::int32_t* next_ids() const noexcept {
    return value<std::int32_t>(decoder_, programs_.decoder.next_ids);
  }

 private:
  // One program's memory plan.
  struct Planned {
    std::vector<std::size_t> bytes;
    MemoryPlan plan;
  };
  // The rows and caches of one step of either program.
  struct Step {
    const StepRow* rows = nullptr;
    const std::int32_t* out_rows = nullptr;
    int n_tokens = 0;
    int n_out = 0;
    const AttnTile* tiles = nullptr;
    int n_tiles = 0;
    const KvPool* self = nullptr;
    const KvPool* cross = nullptr;
    bool suppress = true;
  };

  template <typename T>
  T* value(const Planned& planned, int id) const noexcept {
    return reinterpret_cast<T*>(arena_.as<std::byte>() + planned.plan.offsets[id]);
  }
  void execute(const Program& program, const Planned& planned, const Step& step,
               cudaStream_t stream, const RunHooks* hooks);

  WhisperConfig config_;
  WhisperRules rules_;
  WhisperModelConfig limits_;
  WhisperPrograms programs_;
  Planned encoder_, decoder_;
  DeviceWeights weights_;
  DeviceBuffer arena_;
  DeviceBuffer dequant_;
  // block_table[clip] = clip: the block table of the cache view an encoder step lays over its
  // own q|k|v value.
  DeviceBuffer clip_blocks_;
  DeviceBuffer suppress_always_, suppress_begin_;  // Program::suppress token lists
  Blas blas_;
};

}  // namespace kvb::cuda
