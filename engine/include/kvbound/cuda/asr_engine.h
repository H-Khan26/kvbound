#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <unordered_map>
#include <vector>

#include "kvbound/beam_search.h"
#include "kvbound/block_allocator.h"
#include "kvbound/cuda/device.h"
#include "kvbound/cuda/engine.h"
#include "kvbound/cuda/graph_cache.h"
#include "kvbound/cuda/kv_pool.h"
#include "kvbound/cuda/step_buffers.h"
#include "kvbound/cuda/whisper_model.h"
#include "kvbound/sequence.h"
#include "kvbound/step.h"

namespace kvb::cuda {

// What a speech engine is told besides an EngineConfig. Of that config it reads max_ctx as the
// tokens a decoder sequence may hold, where 0 means the model's max_target_positions, and the
// kv_* fields as the Self cache's; every other field means what it means to Engine.
struct AsrOptions {
  int max_encode_batch = 1;           // clips one encoder step may hold, at most max_seqs
  DType cross_kv_dtype = DType::F16;  // element type of the Cross cache: F16 or I8
  // Beams forked from one parent share its Self blocks, and a block is copied only before a
  // beam writes into it while another beam holds it. Without sharing, no block is shared after
  // a fork: every child but one gets copies of all its parent's blocks. Both give the same
  // tokens and scores; the second leaves nothing for copy-on-write to get wrong.
  bool beam_share_blocks = true;
  // Beam search ranks each beam's candidates on the device (kernels::beam_candidates, in the
  // decoder step's graph), so only n_candidates() pairs per beam cross to the host. Otherwise
  // the logits rows cross and the host ranks them (append_beam_candidates). Both rank the same
  // way; their FP64 sums may round differently in the last bit (kernels::beam_candidates).
  bool beam_topk_device = true;
};

// What beam_transcribe() is told besides the model's rules.
struct BeamOptions {
  int num_beams = 5;  // at most max_seqs
  double length_penalty = 1.0;
  std::int32_t eos_token_id = -1;  // the token a hypothesis ends on; -1 means the rules'
};

// One clip transcribed by beam search.
struct BeamTranscript {
  // The best hypothesis: its generated tokens, the end token included when it ended on one, and
  // its score, the sum of their log-probabilities / length ^ length_penalty.
  std::vector<std::int32_t> ids;
  float score = 0.0f;
  std::vector<BeamHypothesis> hypotheses;  // every kept hypothesis, best first
  // When each decoder step's candidates were on the host, in seconds after the call started.
  // Step i chooses token i of every beam.
  std::vector<double> step_s;
  double encode_s = 0.0;  // the encoder step, waited for
  double seconds = 0.0;   // the whole call
  int block_copies = 0;   // Self blocks copied at forks
};

// Blocks of both caches and block-table rows: how many exist, how many are free, and the sum of
// the reference counts held on them.
struct AsrBlockUsage {
  int self_blocks = 0, self_free = 0, self_refs = 0;
  int cross_blocks = 0, cross_free = 0, cross_refs = 0;
  int rows = 0, rows_free = 0;
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
// they grow. The Cross cache has max_seqs blocks of max_source_positions positions, counted by
// an allocator of their own; each block-table row points at one, block r for row r unless a
// sequence holding the row was given another. An encoder step fills the blocks of the rows it
// is given. encode() and forward() are the raw steps; execute() turns a scheduler's StepPlan
// into them, and beam_transcribe() runs a beam search over them.
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
  AsrBlockUsage block_usage() const;
  const GraphStats& graph_stats() const noexcept { return graphs_.stats(); }
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
  // are produced. A decode-shaped step (every row an output, at most max_seqs rows, no tiles)
  // replays a CUDA graph when use_graph is set; steps with raw_logits have graphs of their own.
  // The caller is responsible for block-table rows (set_blocks), kv_slot values, and for having
  // encoded a clip into the Cross block of every row's seq.
  void forward(std::span<const StepRow> rows, std::span<const std::int32_t> out_rows,
               const ForwardOptions& options = {});
  // Greedy token of each output row of the last decoder step, after the suppress rules unless
  // the step ran with raw_logits.
  std::span<const std::int32_t> next_ids() const noexcept { return buffers_.next_ids(); }
  // Copies the last decoder step's logits [n_out, vocab] to the host: suppressed tokens are
  // -inf unless the step ran with raw_logits. Valid until the next step of either kind.
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
  // --- Beam search ------------------------------------------------------------------------------

  // Transcribes one clip, features [n_mels, n_frames] as for encode(), by beam search under the
  // model's rules (kvb::BeamSearch, Hugging Face's algorithm) on an engine with no sequences.
  // The clip is encoded once into one Cross block, which every beam's block-table row points at.
  // The prompt runs as one beam; then each decoder step runs the live beams as its rows, beam i
  // on the i-th of num_beams rows, and forks their Self blocks as block_allocator.h describes.
  // Hypotheses end on the end token or at max_length tokens (at most max_ctx). Needs
  // num_beams * ceil(max_length / kv_block_size) Self blocks. Everything the call takes is
  // returned to the pools before it returns or throws.
  BeamTranscript beam_transcribe(std::span<const float> features, const BeamOptions& options);

  // Seconds the latest execute() spent in its encoder steps, each waited for; 0 if it had none.
  double encode_seconds() const noexcept { return encode_seconds_; }
  // While set, execute() synchronizes after every op and adds its time to the pass's entry.
  void set_op_times(OpTimes* encoder, OpTimes* decoder) noexcept {
    encoder_times_ = encoder;
    decoder_times_ = decoder;
  }

 private:
  // CachedSequence::blocks are Self blocks.
  struct Sequence : CachedSequence {
    int cross_block = -1;         // held with the row
    bool encoded = false;         // its clip is in its Cross block
    std::vector<float> features;  // [n_mels, n_frames]
  };

  Sequence& sequence(int seq);
  void release_cache(Sequence& s);
  // Points block-table row `row` of the Cross cache at `block`.
  void point_cross(int row, int block);
  // Encodes clip i into Cross block blocks[i].
  void encode_clips(std::span<const std::span<const float>> clips, std::span<const int> blocks,
                    const ForwardOptions& options);
  // forward() that also ranks each output row's beam_k best candidates on the device when
  // beam_k > 0, which needs raw_logits; beam_candidates_on_device() then reads them.
  void decode_step(std::span<const StepRow> rows, std::span<const std::int32_t> out_rows,
                   const ForwardOptions& options, int beam_k);
  // Beam candidates from the last decoder step: copies its logits rows to the host and appends
  // each row's k best (append_beam_candidates).
  void beam_candidates_on_host(std::span<const std::int32_t> suppressed, int k,
                               std::vector<BeamCandidate>& out);
  // Appends the candidates the last decode_step() ranked on the device, k per output row.
  void beam_candidates_on_device(int k, std::vector<BeamCandidate>& out) const;

  EngineConfig config_;
  AsrOptions options_;
  cudaStream_t stream_ = nullptr;
  std::unique_ptr<WhisperModel> model_;
  std::unique_ptr<KvPool> self_;
  std::unique_ptr<KvPool> cross_;
  BlockAllocator blocks_;
  BlockAllocator rows_;
  BlockAllocator cross_blocks_;
  std::vector<int> cross_of_row_;  // the Cross block each block-table row points at
  std::unordered_map<int, Sequence> sequences_;

  // Rows and tiles hold the larger of a decoder step and an encoder step. The features of an
  // encoder step cross as FP32, one row per frame, and are rounded to FP16 on the device.
  StepBuffers buffers_;
  PinnedBuffer mel_host_;
  DeviceBuffer mel_dev_;
  GraphCache graphs_;  // decoder steps, keyed by (n_tokens, n_out)
  // Beam candidates on the device: per token the kernels::kSuppress* flags of the decoding
  // rules, the ranking's scratch, and its output with a pinned host copy, each sized for
  // max_seqs rows of max_seqs + 1 candidates.
  DeviceBuffer beam_suppress_;
  DeviceBuffer beam_scratch_;
  DeviceBuffer beam_out_dev_;
  PinnedBuffer beam_out_host_;
  int beam_max_k_ = 0;
  int beam_out_count_ = 0;  // candidates the latest decode_step() fetched
  double encode_seconds_ = 0.0;
  OpTimes* encoder_times_ = nullptr;
  OpTimes* decoder_times_ = nullptr;
};

}  // namespace kvb::cuda
