#pragma once

#include <cstdint>
#include <functional>
#include <span>
#include <vector>

#include "kvbound/cuda/engine.h"
#include "kvbound/sampler.h"

namespace kvb::cuda {

struct GenerateOptions {
  int max_new_tokens = 16;
  SamplerConfig sampler;
  std::int32_t eos_token_id = -1;  // generation stops after this token; -1 never stops early
  // Called as soon as each token is available.
  std::function<void(std::int32_t)> on_token;
};

// Generates from one prompt on an otherwise idle engine: prefill in chunks of at most
// max_step_tokens, then one token per step. Returns the new tokens only.
std::vector<std::int32_t> generate(Engine& engine, std::span<const std::int32_t> prompt,
                                   const GenerateOptions& options);

}  // namespace kvb::cuda
