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
  double arrival_s = 0.0;  // when the request reaches the queue, in seconds after serve() starts
};

struct ServeOptions {
  SchedulePolicy policy = SchedulePolicy::Static;
  std::int32_t eos_token_id = -1;
};

// One finished step, as seen by an observer.
struct ServeStep {
  const StepPlan& plan;
  const StepResult& result;
  KvUsage usage;   // cache occupancy after the step
  double start_s;  // when the step was planned, in seconds after serve() started
  double time_s;   // when the step finished
};

// Runs every request to completion on an idle engine: the scheduler plans, the engine executes,
// until nothing is left. A request is queued once its arrival time has passed, which is checked
// between steps; when nothing is queued or running the loop waits for the next arrival.
// on_step, when given, is called after each step that processed tokens. If a step throws, every
// sequence of the run is removed from the engine before the exception leaves.
void serve(Engine& engine, std::span<const ServeRequest> requests, const ServeOptions& options = {},
           const std::function<void(const ServeStep&)>& on_step = {});

}  // namespace kvb::cuda
