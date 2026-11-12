#include "kvbound/cuda/serve.h"

#include <map>

namespace kvb::cuda {

void serve(Engine& engine, std::span<const ServeRequest> requests,
           const std::function<void(const ServeStep&)>& on_step, std::int32_t eos_token_id) {
  const EngineConfig& config = engine.config();
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

  while (!scheduler.done()) {
    const StepPlan plan = scheduler.next();
    // A sequence is registered when it is admitted, in the plan that gives it cache blocks.
    for (const StepPlan::Alloc& alloc : plan.alloc) {
      engine.add_sequence(alloc.seq, by_id.at(alloc.seq)->prompt);
    }
    const StepResult result = engine.execute(plan);
    if (plan.run.empty()) continue;
    scheduler.complete(result);
    if (on_step) on_step({plan, result, scheduler.usage()});
  }
}

}  // namespace kvb::cuda
