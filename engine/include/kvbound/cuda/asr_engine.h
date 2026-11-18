#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <unordered_map>
#include <vector>

#include "kvbound/block_allocator.h"
#include "kvbound/cuda/device.h"
#include "kvbound/cuda/engine.h"
#include "kvbound/cuda/kv_pool.h"
#include "kvbound/cuda/whisper_model.h"
#include "kvbound/step.h"

namespace kvb::cuda {

// What a speech engine is told besides an EngineConfig. Of that config it reads max_ctx as the
// tokens a decoder sequence may hold, where 0 means the model's max_target_positions, and the
// kv_* fields as the Self cache's; every other field means what it means to Engine.
struct AsrOptions {
  int max_encode_batch = 1;           // clips one encoder step may hold, at most max_seqs
  DType cross_kv_dtype = DType::F16;  // element type of the Cross cache: F16 or I8
};

// Device memory a speech engine holds, in bytes.
struct AsrMemory {
  std::size_t weights = 0;
  std::size_t self_kv = 0;        // the paged cache of the decoder's own keys and values
  std::size_t cross_kv = 0;       // one block of encoder positions per sequence
  std::size_t arena = 0;          // the activation arena: the larger of the two plans
  std::size_t encoder_arena = 0;  // planned for the largest encoder step
  std::size_t decoder_arena = 0;  // planned for the largest decoder step
  std::size_t dequant = 0;        // scratch for one quantized matrix expanded to FP16
};

// The GPU runtime of a speech model, a sibling of Engine: a Whisper model, its two caches,
// persistent step buffers, and the sequences being transcribed.
//
// The Self cache is Engine's cache: blocks of kv_block_size positions handed to sequences as
// they grow. The Cross cache has one block of max_source_positions positions per block-table
// row, and row r always owns block r; an encoder step fills the blocks of the rows it is given.
// encode() and forward() are the raw steps; execute() turns a scheduler's StepPlan into them.
class AsrEngine {
 public:
  AsrEngine(const std::filesystem::path& model_dir, const EngineConfig& config,
            const AsrOptions& options = {});
  ~AsrEngine();
  AsrEngine(const AsrEngine&) = delete;
  AsrEngine& operator=(const AsrEngine&) = delete;

  const WhisperConfig& model_config() const noexcept { return model_->config(); }
  const WhisperRules& rules() const noexcept { return model_->rules(); }
  const EngineConfig& config() const noexcept { return config_; }
  const AsrOptions& options() const noexcept { return options_; }
  const WhisperPrograms& programs() const noexcept { return model_->programs(); }
  const WeightFormat& weight_format() const noexcept { return model_->weight_format(); }
  AsrMemory memory() const;
  int kv_block_size() const noexcept { return static_cast<int>(self_->dims().block_size); }
  int kv_free_blocks() const noexcept { return blocks_.n_free(); }
  // Elements of one clip's features: n_mels * n_frames.
  std::size_t feature_size() const noexcept;

  // --- Raw steps --------------------------------------------------------------------------------

  // Encodes seqs.size() clips in one encoder step and waits for it. `features` holds their
  // log-mel features end to end, each [n_mels, n_frames] row-major as the feature extractor
  // writes them; clip i fills the Cross block of block-table row seqs[i]. Values are rounded to
  // FP16 and must be finite.
  void encode(std::span<const float> features, std::span<const int> seqs,
              const ForwardOptions& options = {});
  // Runs one decoder step over `rows` and waits for it. out_rows lists the rows whose logits
  // are produced. The caller is responsible for block-table rows (set_blocks), kv_slot values,
  // and for having encoded a clip into the Cross block of every row's seq.
  void forward(std::span<const StepRow> rows, std::span<const std::int32_t> out_rows,
               const ForwardOptions& options = {});
  // Greedy token of each output row of the last decoder step, after the suppress rules.
  std::span<const std::int32_t> next_ids() const noexcept {
    return {next_ids_.as<std::int32_t>(), last_n_out_};
  }
  // Copies the last decoder step's logits [n_out, vocab] to the host: suppressed tokens are
  // -inf. Valid until the next step of either kind.
  std::vector<float> logits();
  // Points block-table row `seq` of the Self cache at these physical blocks.
  void set_blocks(int seq, std::span<const int> blocks);

  // --- Sequences --------------------------------------------------------------------------------

  // Registers a sequence with the features of its clip, [n_mels, n_frames] as for encode(),
  // which are copied. Its tokens start as the prompt of the model's rules. It holds no
  // block-table row and no cache until a plan allocates them.
  void add_sequence(int seq, std::span<const float> features);
  // Applies one plan: frees, preemptions, allocations, the encoder steps of plan.encode (at
  // most max_encode_batch clips each), then one decoder step over the planned tokens. Each
  // sequence whose known tokens are all processed gets its greedy token appended. A preempted
  // sequence keeps its tokens and features but loses both caches, so it must be encoded again.
  StepResult execute(const StepPlan& plan);
  bool has_sequence(int seq) const noexcept { return sequences_.contains(seq); }
  std::size_t n_sequences() const noexcept { return sequences_.size(); }
  // Prompt plus generated tokens of a live sequence.
  const std::vector<std::int32_t>& tokens(int seq) const;
  // Seconds the latest execute() spent in its encoder steps, each waited for; 0 if it had none.
  double encode_seconds() const noexcept { return encode_seconds_; }
  // While set, execute() synchronizes after every op and adds its time to the pass's entry.
  void set_op_times(OpTimes* encoder, OpTimes* decoder) noexcept {
    encoder_times_ = encoder;
    decoder_times_ = decoder;
  }

 private:
  struct Sequence {
    int row = -1;             // block-table row, -1 until blocks are allocated
    std::vector<int> blocks;  // physical Self blocks, in logical order
    std::vector<std::int32_t> tokens;
    int n_cached = 0;             // tokens whose keys and values are in the Self cache
    bool encoded = false;         // its clip is in the Cross block of its row
    std::vector<float> features;  // [n_mels, n_frames]
  };

  Sequence& sequence(int seq);
  void release_cache(Sequence& s);
  void encode_clips(std::span<const std::span<const float>> clips, std::span<const int> seqs,
                    const ForwardOptions& options);

  EngineConfig config_;
  AsrOptions options_;
  cudaStream_t stream_ = nullptr;
  std::unique_ptr<WhisperModel> model_;
  std::unique_ptr<KvPool> self_;
  std::unique_ptr<KvPool> cross_;
  BlockAllocator blocks_;
  BlockAllocator rows_;
  std::unordered_map<int, Sequence> sequences_;

  // Step inputs and outputs: pinned host copies and their persistent device buffers. Rows and
  // tiles hold the larger of a decoder step and an encoder step. The features of an encoder
  // step cross as FP32, one row per frame, and are rounded to FP16 on the device.
  PinnedBuffer rows_host_, out_rows_host_, tiles_host_, next_ids_, mel_host_;
  DeviceBuffer rows_dev_, out_rows_dev_, tiles_dev_, mel_dev_;
  std::size_t last_n_out_ = 0;
  double encode_seconds_ = 0.0;
  OpTimes* encoder_times_ = nullptr;
  OpTimes* decoder_times_ = nullptr;
};

}  // namespace kvb::cuda
