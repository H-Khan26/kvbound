#pragma once

#include <cstdint>
#include <deque>
#include <vector>

#include "kvbound/step.h"

namespace kvb {

// What the scheduler needs to know about a request: lengths, not token ids.
struct Request {
  int id = 0;
  int prompt_tokens = 0;
  int max_new_tokens = 0;
};

struct SchedulerConfig {
  int n_slots = 1;                 // sequences cached at the same time
  int max_ctx = 0;                 // tokens one sequence may hold
  int block_size = 0;              // tokens per KV block; the static design uses max_ctx
  int max_step_tokens = 0;         // rows one step may process
  std::int32_t eos_token_id = -1;  // a sequence ends on this token; -1 runs to max_new_tokens
};

// KV cache occupancy in tokens, after the most recent plan.
struct KvUsage {
  std::int64_t reserved = 0;  // positions in blocks assigned to sequences
  std::int64_t used = 0;      // positions holding a cached token
  int running = 0;            // sequences still generating
  int waiting = 0;            // requests not yet admitted
};

// Static batching: when no batch is running, the next n_slots waiting requests form one. Each
// gets cache blocks for its whole life up front, and no request is admitted until every
// sequence of the batch has finished, so a slot whose sequence ends early idles.
//
// The scheduler holds no device state and never looks at token ids apart from the EOS check; it
// turns request lengths into StepPlans and is driven by next() / complete() pairs.
class Scheduler {
 public:
  explicit Scheduler(const SchedulerConfig& config);

  // Queues a request. Throws kvb::Error if it cannot fit in max_ctx.
  void submit(const Request& request);

  // True when nothing is waiting, running or left to free.
  bool done() const noexcept { return waiting_.empty() && batch_.empty(); }

  // The plan for the next iteration. A plan that runs tokens must be answered with complete()
  // before the next call.
  StepPlan next();
  // Reports the tokens the plan's step produced.
  void complete(const StepResult& result);

  KvUsage usage() const noexcept;
  const SchedulerConfig& config() const noexcept { return config_; }

 private:
  struct Sequence {
    Request request;
    int n_blocks = 0;
    int n_known = 0;   // prompt plus generated tokens
    int n_cached = 0;  // tokens processed into the cache
    int n_generated = 0;
    bool finished = false;
  };

  Sequence& sequence(int id);

  SchedulerConfig config_;
  std::deque<Request> waiting_;
  std::vector<Sequence> batch_;         // the running batch, including members that have finished
  std::vector<StepPlan::Run> planned_;  // rows of the plan awaiting complete()
};

}  // namespace kvb
