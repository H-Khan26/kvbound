#include "kvbound/scheduler.h"

#include <algorithm>
#include <string>

#include "kvbound/error.h"

namespace kvb {

std::string_view schedule_policy_name(SchedulePolicy policy) noexcept {
  return policy == SchedulePolicy::Continuous ? "continuous" : "static";
}

std::optional<SchedulePolicy> schedule_policy_from_name(std::string_view name) noexcept {
  if (name == "static") return SchedulePolicy::Static;
  if (name == "continuous") return SchedulePolicy::Continuous;
  return std::nullopt;
}

Scheduler::Scheduler(const SchedulerConfig& config) : config_(config) {
  if (config.n_slots <= 0 || config.max_ctx <= 0 || config.block_size <= 0 ||
      config.block_size > config.max_ctx) {
    throw Error("scheduler needs n_slots > 0 and 0 < block_size <= max_ctx");
  }
  // Every running sequence must be able to decode one token in the same step.
  if (config.max_step_tokens < config.n_slots) {
    throw Error("max_step_tokens must be at least n_slots");
  }
  const int blocks_per_seq = blocks_for(config.max_ctx);
  if (config_.n_blocks <= 0) config_.n_blocks = config.n_slots * blocks_per_seq;
  const int needed =
      config.policy == SchedulePolicy::Static ? config.n_slots * blocks_per_seq : blocks_per_seq;
  if (config_.n_blocks < needed) {
    throw Error("the KV pool has " + std::to_string(config_.n_blocks) + " blocks; the " +
                std::string(schedule_policy_name(config.policy)) + " policy needs " +
                std::to_string(needed));
  }
  if (config.encode_rows < 0 || (config.encode_rows > 0 && config.max_encodes_per_step < 1)) {
    throw Error("encode_rows must not be negative and encoding needs max_encodes_per_step > 0");
  }
  free_blocks_ = config_.n_blocks;
}

int Scheduler::blocks_for(int n_tokens) const noexcept {
  return (n_tokens + config_.block_size - 1) / config_.block_size;
}

bool Scheduler::can_grow(const Sequence& s) const noexcept {
  return !s.finished && s.n_blocks < blocks_for(s.request.prompt_tokens + s.request.max_new_tokens);
}

void Scheduler::check(const Request& request) const {
  if (request.prompt_tokens <= 0 || request.max_new_tokens <= 0) {
    throw Error("request " + std::to_string(request.id) +
                " needs at least one prompt token and one new token");
  }
  if (request.prompt_tokens + request.max_new_tokens > config_.max_ctx) {
    throw Error("request " + std::to_string(request.id) + " does not fit in max_ctx");
  }
}

void Scheduler::submit(const Request& request) {
  check(request);
  waiting_.push_back({.request = request, .n_known = request.prompt_tokens});
}

Scheduler::Sequence& Scheduler::sequence(int id) {
  const auto it = std::find_if(running_.begin(), running_.end(),
                               [&](const Sequence& s) { return s.request.id == id; });
  if (it == running_.end()) throw Error("unknown sequence " + std::to_string(id));
  return *it;
}

void Scheduler::admit_front(int n_blocks, StepPlan& plan) {
  Sequence s = waiting_.front();
  waiting_.pop_front();
  s.n_blocks = n_blocks;
  free_blocks_ -= n_blocks;
  plan.alloc.push_back({s.request.id, n_blocks});
  running_.push_back(s);
}

void Scheduler::plan_static(StepPlan& plan) {
  const bool batch_finished =
      std::all_of(running_.begin(), running_.end(), [](const Sequence& s) { return s.finished; });
  if (!batch_finished) return;
  for (const Sequence& s : running_) {
    plan.free.push_back(s.request.id);
    free_blocks_ += s.n_blocks;
  }
  running_.clear();
  while (!waiting_.empty() && static_cast<int>(running_.size()) < config_.n_slots) {
    // The whole life of the sequence is reserved at admission.
    const Request& request = waiting_.front().request;
    admit_front(blocks_for(request.prompt_tokens + request.max_new_tokens), plan);
  }
}

void Scheduler::plan_continuous(StepPlan& plan) {
  std::erase_if(running_, [&](const Sequence& s) {
    if (!s.finished) return false;
    plan.free.push_back(s.request.id);
    free_blocks_ += s.n_blocks;
    return true;
  });

  // Growth, oldest sequence first. Victims come off the young end and go to the front of the
  // queue oldest first, so readmission keeps the original order.
  const auto preempt_youngest = [&] {
    Sequence victim = running_.back();
    running_.pop_back();
    plan.preempt.push_back(victim.request.id);
    free_blocks_ += victim.n_blocks;
    victim.n_blocks = 0;
    victim.n_cached = 0;
    victim.encoded = false;
    waiting_.push_front(victim);
  };
  for (std::size_t i = 0; i < running_.size(); ++i) {
    const int need = blocks_for(running_[i].n_known) - running_[i].n_blocks;
    if (need <= 0) continue;
    while (free_blocks_ < need && running_.size() > i + 1) preempt_youngest();
    if (free_blocks_ < need) {
      // The sequence that needs the block is itself the youngest.
      preempt_youngest();
      break;
    }
    running_[i].n_blocks += need;
    free_blocks_ -= need;
    plan.alloc.push_back({running_[i].request.id, need});
  }

  // A plan that had to preempt admits nothing: the pool is full.
  if (!plan.preempt.empty()) return;
  int growing = static_cast<int>(std::count_if(running_.begin(), running_.end(),
                                               [&](const Sequence& s) { return can_grow(s); }));
  while (!waiting_.empty() && static_cast<int>(running_.size()) < config_.n_slots) {
    const Sequence& head = waiting_.front();
    const int need = blocks_for(head.n_known);
    const int total = blocks_for(head.request.prompt_tokens + head.request.max_new_tokens);
    const int spare = growing + (need < total ? 1 : 0);
    // An empty cache admits whatever is first: every request fits in the pool on its own.
    if (!running_.empty() && free_blocks_ < need + spare) break;
    growing = spare;
    admit_front(need, plan);
  }
}

void Scheduler::plan_encodes(StepPlan& plan) {
  if (config_.encode_rows == 0) return;
  for (Sequence& s : running_) {
    if (static_cast<int>(plan.encode.size()) == config_.max_encodes_per_step) break;
    if (s.encoded) continue;
    plan.encode.push_back(s.request.id);
    s.encoded = true;
  }
}

void Scheduler::plan_rows(StepPlan& plan) {
  // A sequence still waiting for the encoder has no cross-attention cache to run against.
  const auto runnable = [&](const Sequence& s) {
    return !s.finished && (config_.encode_rows == 0 || s.encoded);
  };
  // Sequences with one token left to process are decoding and take one row each first; those
  // with more are prefilling and share what is left of the step.
  int budget = config_.max_step_tokens;
  for (const Sequence& s : running_) {
    if (runnable(s) && s.n_known - s.n_cached == 1) {
      plan.run.push_back({s.request.id, 1});
      --budget;
    }
  }
  for (const Sequence& s : running_) {
    const int pending = s.n_known - s.n_cached;
    if (!runnable(s) || pending <= 1 || budget == 0) continue;
    const int n = std::min(pending, budget);
    plan.run.push_back({s.request.id, n});
    budget -= n;
  }
}

StepPlan Scheduler::next() {
  if (!planned_.empty()) throw Error("next() called before complete()");
  StepPlan plan;
  if (config_.policy == SchedulePolicy::Static) {
    plan_static(plan);
  } else {
    plan_continuous(plan);
  }
  plan_encodes(plan);
  plan_rows(plan);
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
  for (const Sequence& s : running_) {
    usage.reserved += static_cast<std::int64_t>(s.n_blocks) * config_.block_size;
    usage.used += s.n_cached;
    usage.running += s.finished ? 0 : 1;
  }
  return usage;
}

}  // namespace kvb
