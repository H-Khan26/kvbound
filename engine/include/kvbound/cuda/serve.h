#pragma once

#include <cstdint>
#include <functional>
#include <span>
#include <vector>

#include "kvbound/cuda/engine.h"
#include "kvbound/sampler.h"
#include "kvbound/scheduler.h"

namespace kvb::cuda {

struct ServeRequest {
  int id = 0;
  std::vector<std::int32_t> prompt;
  int max_new_tokens = 0;
  SamplerConfig sampler;
};

// One finished step, as seen by an observer.
struct ServeStep {
  const StepPlan& plan;
  const StepResult& result;
  KvUsage usage;  // cache occupancy after the step
};

// Runs every request to completion on an idle engine: the scheduler plans, the engine executes,
// until nothing is left. All requests count as queued at the start. on_step, when given, is
// called after each step that processed tokens. If a step throws, every sequence of the run is
// removed from the engine before the exception leaves.
void serve(Engine& engine, std::span<const ServeRequest> requests,
           const std::function<void(const ServeStep&)>& on_step = {},
           std::int32_t eos_token_id = -1);

}  // namespace kvb::cuda
