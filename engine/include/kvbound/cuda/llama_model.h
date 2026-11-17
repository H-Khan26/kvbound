#pragma once

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <vector>

#include "kvbound/cuda/blas.h"
#include "kvbound/cuda/device.h"
#include "kvbound/cuda/kv_pool.h"
#include "kvbound/cuda/weights.h"
#include "kvbound/memory_plan.h"
#include "kvbound/model_config.h"
#include "kvbound/program.h"
#include "kvbound/step.h"

namespace kvb::cuda {

// Wall time spent in each kind of op, measured by synchronizing after every launch.
struct OpTimes {
  std::array<double, kOpKindCount> seconds{};
  std::array<std::int64_t, kOpKindCount> calls{};
};

struct RunHooks {
  // Called for each capture site an op fills, with the program value and its device pointer.
  std::function<void(const std::string& site, int value, const void* data)> on_site;
  // When set, the stream is synchronized after every op and the time is added here.
  OpTimes* times = nullptr;
};

struct LlamaModelConfig {
  int max_ctx = 0;          // positions the RoPE table covers
  int max_step_tokens = 0;  // rows of the largest step
  int max_outputs = 0;      // logit rows of the largest step
  PlanStrategy plan = PlanStrategy::GreedyBySize;
  GemmAccumulate gemm = GemmAccumulate::F32;
  // Build the fused program: stacked q|k|v and gate|up products, residual adds folded into norms,
  // RoPE folded into the cache write.
  bool fuse = true;
  // FP16 matrices are multiplied by kvbound's own kernel for steps of up to this many rows, and
  // by cuBLAS for wider steps; 0 means always cuBLAS.
  int f16_gemv_rows = 4;
  // Read cached keys 16 bytes at a time where the KV layout allows it.
  bool wide_key_loads = true;
  // Quantized matrices are multiplied in place for steps of up to this many rows. Wider steps
  // use tensor cores on the quantized weights, or with expand_quantized, expand each matrix to
  // FP16 first for cuBLAS.
  int quantized_gemm_rows = 16;
  bool expand_quantized = false;
  // An expanded matrix is written this many bytes at a time, as whole output channels, so the
  // vocabulary projection needs no scratch of its own size.
  std::size_t dequant_slice_bytes = std::size_t{32} << 20;
};

// A Llama checkpoint on the device plus everything one forward pass needs: the op program, the
// planned activation arena and the RoPE table. Stateless between steps; the KV cache and the
// step's rows are passed in.
class LlamaModel {
 public:
  LlamaModel(const std::filesystem::path& model_dir, const LlamaModelConfig& config,
             cudaStream_t stream);

  const ModelConfig& config() const noexcept { return config_; }
  const Program& program() const noexcept { return program_; }
  const MemoryPlan& memory_plan() const noexcept { return plan_; }
  const std::vector<std::size_t>& value_sizes() const noexcept { return value_bytes_; }
  std::size_t weight_bytes() const noexcept { return weights_.bytes(); }
  const WeightFormat& weight_format() const noexcept { return weights_.format(); }
  // Scratch that holds one quantized matrix expanded to FP16; empty when no step needs it.
  std::size_t dequant_bytes() const noexcept { return dequant_.bytes(); }
  GemmAccumulate gemm_accumulate() const noexcept { return blas_.accumulate(); }

  // Launches the whole program for one step. rows [n_tokens] and out_rows [n_out] are device
  // arrays. With n_tiles > 0, attention runs the prefill kernel over tiles [n_tiles], a device
  // array of runs of rows (see plan_attention_tiles); otherwise one block per query row. Returns
  // without waiting unless hooks ask for timing or captures.
  void run(const StepRow* rows, const std::int32_t* out_rows, int n_tokens, int n_out,
           const AttnTile* tiles, int n_tiles, const KvPool& kv, cudaStream_t stream,
           const RunHooks* hooks = nullptr);

  // Device outputs of the last run: logits [n_out, vocab] and their argmax [n_out].
  const float* logits() const noexcept { return value<float>(program_.logits); }
  const std::int32_t* next_ids() const noexcept { return value<std::int32_t>(program_.next_ids); }

 private:
  template <typename T>
  T* value(int id) const noexcept {
    return reinterpret_cast<T*>(arena_.as<std::byte>() + plan_.offsets[id]);
  }

  ModelConfig config_;
  LlamaModelConfig limits_;
  Program program_;
  std::vector<std::size_t> value_bytes_;
  MemoryPlan plan_;
  DeviceWeights weights_;
  DeviceBuffer arena_;
  DeviceBuffer dequant_;
  DeviceBuffer rope_table_;
  Blas blas_;
};

}  // namespace kvb::cuda
