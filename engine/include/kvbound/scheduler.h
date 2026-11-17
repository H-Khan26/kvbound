#pragma once

#include <cstdint>
#include <deque>
#include <optional>
#include <string_view>
#include <vector>

#include "kvbound/step.h"

namespace kvb {

// What the scheduler needs to know about a request: lengths, not token ids.
struct Request {
  int id = 0;
  int prompt_tokens = 0;
  int max_new_tokens = 0;
};

enum class SchedulePolicy : std::uint8_t {
  // When no batch is running, the next n_slots waiting requests form one. Each gets cache blocks
  // for its whole life up front, and no request is admitted until every sequence of the batch
  // has finished, so a slot whose sequence ends early idles.
  Static,
  // Requests join and leave between any two steps. A request is admitted when the pool has the
  // blocks its known tokens need, a sequence gets another block when it grows into one, and
  // when the pool runs dry the most recently admitted sequences are preempted.
  Continuous,
};

std::string_view schedule_policy_name(SchedulePolicy policy) noexcept;
std::optional<SchedulePolicy> schedule_policy_from_name(std::string_view name) noexcept;

struct SchedulerConfig {
  SchedulePolicy policy = SchedulePolicy::Static;
  int n_slots = 1;     // sequences cached at the same time
  int max_ctx = 0;     // tokens one sequence may hold
  int block_size = 0;  // tokens per KV block; the static design uses max_ctx
  // Blocks in the pool. 0 means enough for n_slots sequences of max_ctx tokens, which the static
  // policy needs; the continuous policy runs with any pool that holds one such sequence.
  int n_blocks = 0;
  int max_step_tokens = 0;         // rows one step may process
  std::int32_t eos_token_id = -1;  // a sequence ends on this token; -1 runs to max_new_tokens
  // Speech: rows of the encoder pass over one request's audio, 0 when requests carry no audio.
  // The scheduler reads only whether it is above 0: then every sequence is encoded once after
  // each admission, before its first token runs. Encoder rows do not count as step tokens.
  int encode_rows = 0;
  int max_encodes_per_step = 1;  // sequences one plan may encode
};

// KV cache occupancy in tokens, after the most recent plan.
struct KvUsage {
  std::int64_t reserved = 0;  // positions in blocks assigned to sequences
  std::int64_t used = 0;      // positions holding a cached token
  int running = 0;            // sequences holding blocks that are still generating
  int waiting = 0;            // requests not admitted, or preempted and not yet readmitted
};

// Turns request lengths into StepPlans. It holds no device state and never looks at token ids
// apart from the EOS check; it is driven by next() / complete() pairs, and requests may be
// submitted between any two plans.
//
// Continuous policy, in the order one plan is built:
//   1. Finished sequences are freed.
//   2. A decoding sequence about to cross into a new block gets one. If none is free, the most
//      recently admitted sequences are preempted until one is: they return to the front of the
//      queue and are recomputed from their first token when readmitted. The oldest running
//      sequence is never preempted, so the queue always drains.
//   3. Waiting requests are admitted first come, first served, while a block-table row is free
//      and the pool holds the blocks for the request's known tokens plus one spare block for
//      every running sequence that can still grow. The spare blocks let each of them decode at
//      least block_size tokens before anything has to be preempted.
//   4. Every decoding sequence gets one row; sequences still prefilling share the rest of the
//      step's rows in admission order.
//
// With encode_rows set, under either policy, a plan also encodes the first max_encodes_per_step
// running sequences that are not encoded yet, in admission order, and gives rows only to encoded
// sequences. Admission is unchanged, so a static batch still fills at once and its members
// start a few plans apart. Preemption drops the encoding with the blocks: a readmitted sequence
// is encoded again.
class Scheduler {
 public:
  explicit Scheduler(const SchedulerConfig& config);

  // Throws kvb::Error if the request is empty or cannot fit in max_ctx.
  void check(const Request& request) const;
  // Queues a request that passes check().
  void submit(const Request& request);

  // True when nothing is waiting, running or left to free.
  bool done() const noexcept { return waiting_.empty() && running_.empty(); }

  // The plan for the next iteration. A plan that runs tokens must be answered with complete()
  // before the next call.
  StepPlan next();
  // Reports the tokens the plan's step produced.
  void complete(const StepResult& result);

  KvUsage usage() const noexcept;
  int free_blocks() const noexcept { return free_blocks_; }
  const SchedulerConfig& config() const noexcept { return config_; }

 private:
  struct Sequence {
    Request request;
    int n_blocks = 0;
    int n_known = 0;   // prompt plus generated tokens
    int n_cached = 0;  // tokens processed into the cache
    int n_generated = 0;
    bool finished = false;
    bool encoded = false;  // its audio went through the encoder since it was last admitted
  };

  Sequence& sequence(int id);
  int blocks_for(int n_tokens) const noexcept;
  bool can_grow(const Sequence& s) const noexcept;
  void admit_front(int n_blocks, StepPlan& plan);  // the first waiting request starts running
  void plan_static(StepPlan& plan);
  void plan_continuous(StepPlan& plan);
  void plan_encodes(StepPlan& plan);
  void plan_rows(StepPlan& plan);

  SchedulerConfig config_;
  int free_blocks_ = 0;
  std::deque<Sequence> waiting_;  // arrival order, preempted sequences first
  // Sequences holding blocks, in admission order. Under the static policy this is the running
  // batch, including members that have finished.
  std::vector<Sequence> running_;
  std::vector<StepPlan::Run> planned_;  // rows of the plan awaiting complete()
};

}  // namespace kvb
