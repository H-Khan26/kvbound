#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "kvbound/block_allocator.h"
#include "kvbound/cuda/device.h"
#include "kvbound/cuda/kv_pool.h"
#include "kvbound/cuda/llama_model.h"
#include "kvbound/sampler.h"
#include "kvbound/step.h"

namespace kvb::cuda {

struct EngineConfig {
  int max_ctx = 2048;       // tokens one sequence can hold
  int max_seqs = 1;         // sequences cached at the same time
  int max_step_tokens = 0;  // rows of the largest step; 0 means max_ctx
  int kv_block_size = 0;    // positions per KV block; 0 means max_ctx, one block per sequence
  int kv_blocks = 0;        // blocks in the pool; 0 means enough for max_seqs full sequences
  KvLayout kv_layout = KvLayout::HeadMajor;
  PlanStrategy plan = PlanStrategy::GreedyBySize;
  GemmAccumulate gemm = GemmAccumulate::F32;
  // Read cached keys 16 bytes at a time where the KV layout allows it.
  bool wide_key_loads = true;
  // Replays decode-shaped steps from a captured CUDA graph instead of launching each kernel.
  bool use_graph = true;
};

// Capture-site name to values, row-major with one row per token (or per output), as FP32.
using Capture = std::map<std::string, std::vector<float>>;

struct ForwardOptions {
  Capture* capture = nullptr;
  OpTimes* times = nullptr;
};

// Device memory the engine holds, in bytes.
struct EngineMemory {
  std::size_t weights = 0;
  std::size_t kv_pool = 0;
  std::size_t arena = 0;            // planned activation arena
  std::size_t arena_naive = 0;      // the same values without sharing
  std::size_t arena_peak_live = 0;  // lower bound for any plan
};

// The GPU runtime: model, KV cache, persistent step buffers, and the sequences being generated.
// forward() is the raw step; execute() turns a scheduler's StepPlan into one.
class Engine {
 public:
  Engine(const std::filesystem::path& model_dir, const EngineConfig& config);
  ~Engine();
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  const ModelConfig& model_config() const noexcept { return model_->config(); }
  const EngineConfig& config() const noexcept { return config_; }
  const Program& program() const noexcept { return model_->program(); }
  EngineMemory memory() const;
  // Where each program value lives in the activation arena, and its size at the largest step.
  const MemoryPlan& memory_plan() const noexcept { return model_->memory_plan(); }
  const std::vector<std::size_t>& value_sizes() const noexcept { return model_->value_sizes(); }
  int kv_block_size() const noexcept { return static_cast<int>(kv_->dims().block_size); }
  int kv_free_blocks() const noexcept { return blocks_.n_free(); }

  // --- Raw step -------------------------------------------------------------------------------

  // Runs one step over `rows` and waits for it. out_rows lists the rows whose logits are
  // produced. The caller is responsible for block-table rows (set_blocks) and kv_slot values.
  void forward(std::span<const StepRow> rows, std::span<const std::int32_t> out_rows,
               const ForwardOptions& options = {});
  // Greedy token of each output row of the last step.
  std::span<const std::int32_t> next_ids() const noexcept {
    return {next_ids_.as<std::int32_t>(), last_n_out_};
  }
  // Copies the last step's logits [n_out, vocab] to the host.
  std::vector<float> logits();
  // Points block-table row `seq` at these physical blocks.
  void set_blocks(int seq, std::span<const int> blocks);

  // --- Sequences --------------------------------------------------------------------------------

  // Registers a sequence with its prompt. It holds no block-table row and no cache blocks until
  // a plan allocates them, so more sequences can be registered than fit in the cache at once.
  void add_sequence(int seq, std::span<const std::int32_t> prompt,
                    const SamplerConfig& sampler = {});
  // Applies one plan: frees, allocations, then a forward step over the planned tokens. Each
  // sequence whose known tokens are all processed gets a sampled token appended.
  StepResult execute(const StepPlan& plan);
  // Prompt plus generated tokens of a live sequence.
  const std::vector<std::int32_t>& tokens(int seq) const;
  // While set, execute() synchronizes after every op and adds its time here.
  void set_op_times(OpTimes* times) noexcept { op_times_ = times; }

 private:
  struct Sequence {
    int row = -1;             // block-table row, -1 until blocks are allocated
    std::vector<int> blocks;  // physical blocks, in logical order
    std::vector<std::int32_t> tokens;
    int n_cached = 0;  // tokens whose keys and values are in the cache
    Sampler sampler;
  };

  Sequence& sequence(int seq);
  void run_graph(int n_tokens, int n_out);

  EngineConfig config_;
  cudaStream_t stream_ = nullptr;
  std::unique_ptr<LlamaModel> model_;
  std::unique_ptr<KvPool> kv_;
  BlockAllocator blocks_;
  BlockAllocator rows_;
  std::unordered_map<int, Sequence> sequences_;

  // Step inputs and outputs: pinned host copies and their persistent device buffers.
  PinnedBuffer rows_host_, out_rows_host_, next_ids_;
  DeviceBuffer rows_dev_, out_rows_dev_;
  std::size_t last_n_out_ = 0;
  OpTimes* op_times_ = nullptr;
  std::unordered_map<std::int64_t, cudaGraphExec_t> graphs_;  // keyed by (n_tokens, n_out)
};

}  // namespace kvb::cuda
