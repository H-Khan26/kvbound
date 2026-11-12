#include "kvbound/cuda/generate.h"

#include "kvbound/cuda/serve.h"

namespace kvb::cuda {

std::vector<std::int32_t> generate(Engine& engine, std::span<const std::int32_t> prompt,
                                   const GenerateOptions& options) {
  if (options.max_new_tokens <= 0) return {};
  const int total = static_cast<int>(prompt.size()) + options.max_new_tokens;
  if (total > engine.config().max_ctx) throw Error("prompt plus new tokens exceed max_ctx");

  // One request through the same scheduler that plans batches.
  const ServeRequest request{.id = 0,
                             .prompt = {prompt.begin(), prompt.end()},
                             .max_new_tokens = options.max_new_tokens,
                             .sampler = options.sampler};
  std::vector<std::int32_t> generated;
  serve(
      engine, {&request, 1},
      [&](const ServeStep& step) {
        for (const StepResult::Token& token : step.result.tokens) {
          generated.push_back(token.token);
          if (options.on_token) options.on_token(token.token);
        }
      },
      options.eos_token_id);
  return generated;
}

}  // namespace kvb::cuda
