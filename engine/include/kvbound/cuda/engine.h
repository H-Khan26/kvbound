#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "kvbound/block_allocator.h"
#include "kvbound/cuda/device.h"
#include "kvbound/cuda/graph_cache.h"
#include "kvbound/cuda/kv_pool.h"
#include "kvbound/cuda/llama_model.h"
#include "kvbound/cuda/run_hooks.h"
#include "kvbound/cuda/step_buffers.h"
#include "kvbound/sampler.h"
#include "kvbound/sequence.h"
#include "kvbound/step.h"

namespace kvb::cuda {

struct EngineConfig {
  int max_ctx = 2048;       // tokens one sequence can hold
  int max_seqs = 1;         // sequences cached at the same time
  int max_step_tokens = 0;  // rows of the largest step; 0 means max_ctx
  // Rows of one step that can produce logits; 0 means max_seqs, one per sequence, which is all
  // that generation needs. Logits are the widest activation, so this sets the arena's size.
  int max_outputs = 0;
  int kv_block_size = 0;  // positions per KV block; 0 means max_ctx, one block per sequence
  int kv_blocks = 0;      // blocks in the pool; 0 means enough for max_seqs full sequences
  KvLayout kv_layout = KvLayout::HeadMajor;
  DType kv_dtype = DType::F16;  // F16, or I8 with one scale per cached vector
  PlanStrategy plan = PlanStrategy::GreedyBySize;
  GemmAccumulate gemm = GemmAccumulate::F32;
  // The fused program, and the row count up to which FP16 products use kvbound's own kernel
  // instead of cuBLAS (0: always cuBLAS). See LinearOptions.
  bool fuse = true;
  int f16_gemv_rows = 4;
  // Read cached keys 16 bytes at a time where the KV layout allows it.
  bool wide_key_loads = true;
  // Steps holding several rows of one sequence (prefill) run attention as tiles of rows on
  // tensor cores; otherwise every query row is a block of the decode kernel. In a step that
  // mixes a prompt chunk with decode rows, each decode row is a tile of its own, which leaves
  // most of its block idle.
  bool prefill_attention = true;
  // Steps of up to this many rows multiply quantized weights in place; wider steps use tensor
  // cores on the quantized weights, or with expand_quantized, expand each matrix to FP16 first.
  int quantized_gemm_rows = 3;
  bool expand_quantized = false;
  // Bytes of FP16 a quantized matrix is expanded to at a time, as whole output channels.
  std::size_t dequant_slice_bytes = std::size_t{32} << 20;
  // Replays decode-shaped steps from a captured CUDA graph instead of launching each kernel.
  bool use_graph = true;
  // Graphs kept at once. A graph serves one row count, and continuous batching changes the row
  // count whenever a sequence joins or leaves, so the least recently used graph is dropped when
  // a new row count needs one.
  int max_graphs = 16;
};

// The config with every field left at 0 filled in. Throws kvb::Error on limits that do not fit
// together.
EngineConfig resolved(EngineConfig config);

struct ForwardOptions {
  Capture* capture = nullptr;
  OpTimes* times = nullptr;
  // Speech decoder steps only: skip the suppress rules, so the logits and their argmax are the
  // model's as computed. Engine has no suppress rules and ignores it.
  bool raw_logits = false;
};

// Device memory the engine holds, in bytes.
struct EngineMemory {
  std::size_t weights = 0;
  std::size_t kv_pool = 0;
  std::size_t arena = 0;            // planned activation arena
  std::size_t arena_naive = 0;      // the same values without sharing
  std::size_t arena_peak_live = 0;  // lower bound for any plan
  std::size_t dequant = 0;          // scratch for one quantized matrix expanded to FP16
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
  const WeightFormat& weight_format() const noexcept { return model_->weight_format(); }
  EngineMemory memory() const;
  // Where each program value lives in the activation arena, and its size at the largest step.
  const MemoryPlan& memory_plan() const noexcept { return model_->memory_plan(); }
  const std::vector<std::size_t>& value_sizes() const noexcept { return model_->value_sizes(); }
  int kv_block_size() const noexcept { return static_cast<int>(kv_->dims().block_size); }
  int kv_free_blocks() const noexcept { return blocks_.n_free(); }
  const GraphStats& graph_stats() const noexcept { return graphs_.stats(); }

  // --- Raw step -------------------------------------------------------------------------------

  // Runs one step over `rows` and waits for it. out_rows lists the rows whose logits are
  // produced. The caller is responsible for block-table rows (set_blocks) and kv_slot values.
  void forward(std::span<const StepRow> rows, std::span<const std::int32_t> out_rows,
               const ForwardOptions& options = {});
  // Greedy token of each output row of the last step.
  std::span<const std::int32_t> next_ids() const noexcept { return buffers_.next_ids(); }
  // Copies the last step's logits [n_out, vocab] to the host.
  std::vector<float> logits();
  // Points block-table row `seq` at these physical blocks.
  void set_blocks(int seq, std::span<const int> blocks);

  // --- Sequences --------------------------------------------------------------------------------

  // Registers a sequence with its prompt. It holds no block-table row and no cache blocks until
  // a plan allocates them, so more sequences can be registered than fit in the cache at once.
  void add_sequence(int seq, std::span<const std::int32_t> prompt,
                    const SamplerConfig& sampler = {});
  // Applies one plan: frees, preemptions, allocations, then a forward step over the planned
  // tokens. Each sequence whose known tokens are all processed gets a sampled token appended. A
  // preempted sequence keeps its tokens and sampler but holds no cache until blocks are
  // allocated again.
  StepResult execute(const StepPlan& plan);
  bool has_sequence(int seq) const noexcept { return sequences_.contains(seq); }
  std::size_t n_sequences() const noexcept { return sequences_.size(); }
  // Prompt plus generated tokens of a live sequence.
  const std::vector<std::int32_t>& tokens(int seq) const;
  // While set, execute() synchronizes after every op and adds its time here.
  void set_op_times(OpTimes* times) noexcept { op_times_ = times; }

 private:
  struct Sequence : CachedSequence {
    Sampler sampler;
  };

  Sequence& sequence(int seq);

  EngineConfig config_;
  cudaStream_t stream_ = nullptr;
  std::unique_ptr<LlamaModel> model_;
  std::unique_ptr<KvPool> kv_;
  BlockAllocator blocks_;
  BlockAllocator rows_;
  std::unordered_map<int, Sequence> sequences_;

  StepBuffers buffers_;
  OpTimes* op_times_ = nullptr;
  GraphCache graphs_;  // keyed by (n_tokens, n_out)
};

}  // namespace kvb::cuda
