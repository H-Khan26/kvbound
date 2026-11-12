#include "kvbound/cuda/generate.h"

#include <algorithm>

namespace kvb::cuda {

std::vector<std::int32_t> generate(Engine& engine, std::span<const std::int32_t> prompt,
                                   const GenerateOptions& options) {
  constexpr int kSeq = 0;
  const int total = static_cast<int>(prompt.size()) + options.max_new_tokens;
  if (total > engine.config().max_ctx) throw Error("prompt plus new tokens exceed max_ctx");
  if (options.max_new_tokens <= 0) return {};

  engine.add_sequence(kSeq, prompt, options.sampler);
  const int block_size = engine.kv_block_size();
  StepPlan plan;
  plan.alloc.push_back({kSeq, (total + block_size - 1) / block_size});

  StepPlan release;
  release.free.push_back(kSeq);
  std::vector<std::int32_t> generated;
  try {
    int pending = static_cast<int>(prompt.size());  // known tokens not yet in the cache
    while (static_cast<int>(generated.size()) < options.max_new_tokens) {
      const int n = std::min(pending, engine.config().max_step_tokens);
      plan.run.assign(1, {kSeq, n});
      const StepResult result = engine.execute(plan);
      plan.alloc.clear();
      pending -= n;
      if (result.tokens.empty()) continue;  // a prompt chunk that was not the last

      const std::int32_t token = result.tokens.front().token;
      generated.push_back(token);
      pending = 1;
      if (options.on_token) options.on_token(token);
      if (token == options.eos_token_id) break;
    }
  } catch (...) {
    // The engine stays usable after a failed request.
    engine.execute(release);
    throw;
  }
  engine.execute(release);
  return generated;
}

}  // namespace kvb::cuda
