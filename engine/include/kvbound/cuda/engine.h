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
  // instead of cuBLAS (0: always cuBLAS). See LlamaModelConfig.
  bool fuse = true;
  int f16_gemv_rows = 4;
  // Read cached keys 16 bytes at a time where the KV layout allows it.
  bool wide_key_loads = true;
  // Steps of up to this many rows multiply quantized weights in place; wider steps use tensor
  // cores on the quantized weights, or with expand_quantized, expand each matrix to FP16 first.
  int quantized_gemm_rows = 16;
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

// How the engine launched its steps so far.
struct GraphStats {
  std::int64_t replayed = 0;  // steps launched from a graph
  std::int64_t eager = 0;     // steps launched kernel by kernel
  int captured = 0;           // graphs built, counting a row count again after its eviction
  int evicted = 0;
  double capture_s = 0.0;  // time spent capturing and instantiating
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
  const GraphStats& graph_stats() const noexcept { return graph_stats_; }

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
  struct Sequence {
    int row = -1;             // block-table row, -1 until blocks are allocated
    std::vector<int> blocks;  // physical blocks, in logical order
    std::vector<std::int32_t> tokens;
    int n_cached = 0;  // tokens whose keys and values are in the cache
    Sampler sampler;
  };

  struct Graph {
    cudaGraphExec_t exec = nullptr;
    std::int64_t last_used = 0;  // value of graph_stats_.replayed at the latest launch
  };

  Sequence& sequence(int seq);
  void release_cache(Sequence& s);
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
  std::unordered_map<std::int64_t, Graph> graphs_;  // keyed by (n_tokens, n_out)
  GraphStats graph_stats_;
};

}  // namespace kvb::cuda
