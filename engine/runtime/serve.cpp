#include "kvbound/cuda/serve.h"

#include <map>
#include <vector>

namespace kvb::cuda {

void serve(Engine& engine, std::span<const ServeRequest> requests,
           const std::function<void(const ServeStep&)>& on_step, std::int32_t eos_token_id) {
  const EngineConfig& config = engine.config();
  const int blocks_per_seq = (config.max_ctx + config.kv_block_size - 1) / config.kv_block_size;
  if (engine.n_sequences() != 0) throw Error("serve() needs an idle engine");
  if (config.kv_blocks < config.max_seqs * blocks_per_seq) {
    throw Error("the KV pool cannot hold max_seqs full sequences");
  }
  Scheduler scheduler({.n_slots = config.max_seqs,
                       .max_ctx = config.max_ctx,
                       .block_size = config.kv_block_size,
                       .max_step_tokens = config.max_step_tokens,
                       .eos_token_id = eos_token_id});
  std::map<int, const ServeRequest*> by_id;
  for (const ServeRequest& request : requests) {
    if (!by_id.emplace(request.id, &request).second) throw Error("duplicate request id");
    scheduler.submit({.id = request.id,
                      .prompt_tokens = static_cast<int>(request.prompt.size()),
                      .max_new_tokens = request.max_new_tokens});
  }

  std::vector<int> registered;
  try {
    while (!scheduler.done()) {
      const StepPlan plan = scheduler.next();
      // A sequence is registered when it is admitted, in the plan that gives it cache blocks.
      for (const StepPlan::Alloc& alloc : plan.alloc) {
        const ServeRequest& request = *by_id.at(alloc.seq);
        engine.add_sequence(alloc.seq, request.prompt, request.sampler);
        registered.push_back(alloc.seq);
      }
      const StepResult result = engine.execute(plan);
      if (plan.run.empty()) continue;
      scheduler.complete(result);
      if (on_step) on_step({plan, result, scheduler.usage()});
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
