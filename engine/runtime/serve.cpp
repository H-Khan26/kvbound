#include "kvbound/cuda/serve.h"

#include <algorithm>
#include <chrono>
#include <map>
#include <thread>
#include <vector>

namespace kvb::cuda {
namespace {

using Clock = std::chrono::steady_clock;

// Sleeps until `deadline`. The operating system may wake a sleeper several milliseconds late, so
// the last stretch is spent yielding instead.
void wait_until(Clock::time_point deadline) {
  constexpr auto kSpin = std::chrono::milliseconds(20);
  const auto remaining = deadline - Clock::now();
  if (remaining > kSpin) std::this_thread::sleep_for(remaining - kSpin);
  while (Clock::now() < deadline) std::this_thread::yield();
}

Request lengths(const ServeRequest& request) {
  return {.id = request.id,
          .prompt_tokens = static_cast<int>(request.prompt.size()),
          .max_new_tokens = request.max_new_tokens};
}

}  // namespace

void serve(Engine& engine, std::span<const ServeRequest> requests, const ServeOptions& options,
           const std::function<void(const ServeStep&)>& on_step) {
  const EngineConfig& config = engine.config();
  if (engine.n_sequences() != 0) throw Error("serve() needs an idle engine");
  Scheduler scheduler({.policy = options.policy,
                       .n_slots = config.max_seqs,
                       .max_ctx = config.max_ctx,
                       .block_size = config.kv_block_size,
                       .n_blocks = config.kv_blocks,
                       .max_step_tokens = config.max_step_tokens,
                       .eos_token_id = options.eos_token_id});

  // Every request is checked before any runs, so a bad one fails the call up front.
  std::map<int, const ServeRequest*> by_id;
  std::vector<const ServeRequest*> arrivals;
  for (const ServeRequest& request : requests) {
    if (!by_id.emplace(request.id, &request).second) throw Error("duplicate request id");
    if (!(request.arrival_s >= 0.0)) throw Error("arrival times must be >= 0");
    scheduler.check(lengths(request));
    arrivals.push_back(&request);
  }
  std::stable_sort(
      arrivals.begin(), arrivals.end(),
      [](const ServeRequest* a, const ServeRequest* b) { return a->arrival_s < b->arrival_s; });

  std::vector<int> registered;
  std::size_t n_arrived = 0;
  const Clock::time_point start = Clock::now();
  const auto elapsed = [&] { return std::chrono::duration<double>(Clock::now() - start).count(); };
  try {
    for (;;) {
      const double now = elapsed();
      while (n_arrived < arrivals.size() && arrivals[n_arrived]->arrival_s <= now) {
        scheduler.submit(lengths(*arrivals[n_arrived++]));
      }
      if (scheduler.done()) {
        if (n_arrived == arrivals.size()) break;
        wait_until(start + std::chrono::duration_cast<Clock::duration>(
                               std::chrono::duration<double>(arrivals[n_arrived]->arrival_s)));
        continue;
      }
      const StepPlan plan = scheduler.next();
      // A sequence is registered when it is first admitted. A readmitted one is still known.
      for (const StepPlan::Alloc& alloc : plan.alloc) {
        if (engine.has_sequence(alloc.seq)) continue;
        const ServeRequest& request = *by_id.at(alloc.seq);
        engine.add_sequence(alloc.seq, request.prompt, request.sampler);
        registered.push_back(alloc.seq);
      }
      const StepResult result = engine.execute(plan);
      if (plan.run.empty()) continue;
      scheduler.complete(result);
      if (on_step) on_step({plan, result, scheduler.usage(), now, elapsed()});
    }
  } catch (...) {
    // The engine stays usable after a failed run: nothing this run registered remains.
    StepPlan release;
    for (const int seq : registered) {
      if (engine.has_sequence(seq)) release.free.push_back(seq);
    }
    engine.execute(release);
    throw;
  }
}

}  // namespace kvb::cuda
