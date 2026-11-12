#include "kvbound/scheduler.h"

#include <algorithm>
#include <string>

#include "kvbound/error.h"

namespace kvb {

Scheduler::Scheduler(const SchedulerConfig& config) : config_(config) {
  if (config.n_slots <= 0 || config.max_ctx <= 0 || config.block_size <= 0 ||
      config.block_size > config.max_ctx) {
    throw Error("scheduler needs n_slots > 0 and 0 < block_size <= max_ctx");
  }
  // Every running sequence must be able to decode one token in the same step.
  if (config.max_step_tokens < config.n_slots) {
    throw Error("max_step_tokens must be at least n_slots");
  }
}

void Scheduler::submit(const Request& request) {
  if (request.prompt_tokens <= 0 || request.max_new_tokens <= 0) {
    throw Error("request " + std::to_string(request.id) +
                " needs at least one prompt token and one new token");
  }
  if (request.prompt_tokens + request.max_new_tokens > config_.max_ctx) {
    throw Error("request " + std::to_string(request.id) + " does not fit in max_ctx");
  }
  waiting_.push_back(request);
}

Scheduler::Sequence& Scheduler::sequence(int id) {
  const auto it = std::find_if(batch_.begin(), batch_.end(),
                               [&](const Sequence& s) { return s.request.id == id; });
  if (it == batch_.end()) throw Error("unknown sequence " + std::to_string(id));
  return *it;
}

StepPlan Scheduler::next() {
  if (!planned_.empty()) throw Error("next() called before complete()");
  StepPlan plan;

  const bool batch_finished =
      std::all_of(batch_.begin(), batch_.end(), [](const Sequence& s) { return s.finished; });
  if (batch_finished) {
    for (const Sequence& s : batch_) plan.free.push_back(s.request.id);
    batch_.clear();
    while (!waiting_.empty() && static_cast<int>(batch_.size()) < config_.n_slots) {
      const Request request = waiting_.front();
      waiting_.pop_front();
      // The whole life of the sequence is reserved at admission.
      const int total = request.prompt_tokens + request.max_new_tokens;
      const int n_blocks = (total + config_.block_size - 1) / config_.block_size;
      batch_.push_back(
          {.request = request, .n_blocks = n_blocks, .n_known = request.prompt_tokens});
      plan.alloc.push_back({request.id, n_blocks});
    }
  }

  // Decoding sequences take one row each first; prefilling ones share what is left of the step.
  int budget = config_.max_step_tokens;
  for (const Sequence& s : batch_) {
    if (!s.finished && s.n_generated > 0) {
      plan.run.push_back({s.request.id, 1});
      --budget;
    }
  }
  for (const Sequence& s : batch_) {
    if (s.finished || s.n_generated > 0 || budget == 0) continue;
    const int n = std::min(s.n_known - s.n_cached, budget);
    plan.run.push_back({s.request.id, n});
    budget -= n;
  }
  planned_ = plan.run;
  return plan;
}

void Scheduler::complete(const StepResult& result) {
  if (planned_.empty()) throw Error("complete() called without a running plan");
  for (const StepPlan::Run& run : planned_) sequence(run.seq).n_cached += run.n_tokens;
  planned_.clear();

  for (const StepResult::Token& token : result.tokens) {
    Sequence& s = sequence(token.seq);
    if (s.finished || s.n_cached != s.n_known) {
      throw Error("unexpected token for sequence " + std::to_string(token.seq));
    }
    s.n_known += 1;
    s.n_generated += 1;
    s.finished = s.n_generated == s.request.max_new_tokens || token.token == config_.eos_token_id;
  }
}

KvUsage Scheduler::usage() const noexcept {
  KvUsage usage;
  usage.waiting = static_cast<int>(waiting_.size());
  for (const Sequence& s : batch_) {
    usage.reserved += static_cast<std::int64_t>(s.n_blocks) * config_.block_size;
    usage.used += s.n_cached;
    usage.running += s.finished ? 0 : 1;
  }
  return usage;
}

}  // namespace kvb
