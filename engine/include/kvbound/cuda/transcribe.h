#pragma once

#include <functional>
#include <span>

#include "kvbound/cuda/asr_engine.h"
#include "kvbound/scheduler.h"

namespace kvb::cuda {

struct TranscribeRequest {
  int id = 0;
  std::span<const float> features;  // [n_mels, n_frames], as for AsrEngine::encode()
};

struct TranscribeOptions {
  SchedulePolicy policy = SchedulePolicy::Static;
};

// One finished plan, as seen by an observer.
struct TranscribeStep {
  const StepPlan& plan;
  const StepResult& result;
  KvUsage usage;    // Self cache occupancy after the step
  double start_s;   // when the plan was made, in seconds after transcribe() started
  double encode_s;  // seconds the plan's encoder steps took, each waited for; 0 without any
  double time_s;    // when the step finished
};

// Transcribes every request greedily on an idle engine under the model's rules: each sequence
// starts as the rules' prompt and grows until it emits eos_token_id or holds max_length tokens
// (at most the engine's max_ctx). A Scheduler with encode_rows set plans the steps, so every
// admitted sequence is encoded, at most max_encode_batch per plan, before its first decoder row,
// and a preempted one is encoded again when readmitted. on_step, when given, is called after
// each plan that encoded or decoded. If a step throws, every sequence of the run is removed from
// the engine before the exception leaves.
void transcribe(AsrEngine& engine, std::span<const TranscribeRequest> requests,
                const TranscribeOptions& options = {},
                const std::function<void(const TranscribeStep&)>& on_step = {});

}  // namespace kvb::cuda
