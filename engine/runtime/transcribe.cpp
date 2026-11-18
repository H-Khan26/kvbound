#include "kvbound/cuda/transcribe.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <string>
#include <vector>

namespace kvb::cuda {

void transcribe(AsrEngine& engine, std::span<const TranscribeRequest> requests,
                const TranscribeOptions& options,
                const std::function<void(const TranscribeStep&)>& on_step) {
  using Clock = std::chrono::steady_clock;
  if (engine.n_sequences() != 0) throw Error("transcribe() needs an idle engine");
  const EngineConfig& config = engine.config();
  const WhisperRules& rules = engine.rules();
  const int prompt = static_cast<int>(rules.prompt.size());
  const int max_length = static_cast<int>(std::min<std::int64_t>(rules.max_length, config.max_ctx));
  if (max_length <= prompt) throw Error("max_ctx leaves no room after the prompt");
  Scheduler scheduler({.policy = options.policy,
                       .n_slots = config.max_seqs,
                       .max_ctx = config.max_ctx,
                       .block_size = config.kv_block_size,
                       .n_blocks = config.kv_blocks,
                       .max_step_tokens = config.max_step_tokens,
                       .eos_token_id = rules.eos_token_id,
                       .encode_rows = static_cast<int>(engine.model_config().max_source_positions),
                       .max_encodes_per_step = engine.options().max_encode_batch});

  // Every request is checked before any runs, so a bad one fails the call up front.
  std::map<int, const TranscribeRequest*> by_id;
  for (const TranscribeRequest& request : requests) {
    if (!by_id.emplace(request.id, &request).second) throw Error("duplicate request id");
    if (request.features.size() != engine.feature_size()) {
      throw Error("request " + std::to_string(request.id) + " has features of the wrong size");
    }
    // The encoder rounds features to FP16, as AsrEngine::encode() checks again.
    if (!std::all_of(request.features.begin(), request.features.end(),
                     [](float value) { return std::abs(value) <= 65504.0f; })) {
      throw Error("request " + std::to_string(request.id) + ": features must be finite in FP16");
    }
    scheduler.submit(
        {.id = request.id, .prompt_tokens = prompt, .max_new_tokens = max_length - prompt});
  }

  std::vector<int> registered;
  const Clock::time_point start = Clock::now();
  const auto elapsed = [&] { return std::chrono::duration<double>(Clock::now() - start).count(); };
  try {
    while (!scheduler.done()) {
      const double now = elapsed();
      const StepPlan plan = scheduler.next();
      // A sequence is registered when it is first admitted. A readmitted one is still known.
      for (const StepPlan::Alloc& alloc : plan.alloc) {
        if (engine.has_sequence(alloc.seq)) continue;
        engine.add_sequence(alloc.seq, by_id.at(alloc.seq)->features);
        registered.push_back(alloc.seq);
      }
      const StepResult result = engine.execute(plan);
      if (!plan.run.empty()) scheduler.complete(result);
      if (on_step && (!plan.run.empty() || !plan.encode.empty())) {
        on_step({plan, result, scheduler.usage(), now, engine.encode_seconds(), elapsed()});
      }
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
