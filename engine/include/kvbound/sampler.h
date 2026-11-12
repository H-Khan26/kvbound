#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace kvb {

struct SamplerConfig {
  float temperature = 0.0f;  // 0 selects the largest logit (greedy)
  int top_k = 0;             // keep the k largest logits; 0 keeps all
  float top_p = 1.0f;        // keep the smallest set of tokens whose probability reaches top_p
  std::uint64_t seed = 0;

  bool greedy() const noexcept { return temperature <= 0.0f; }
};

// Picks the next token from one row of logits on the CPU. Filters apply in the order
// temperature, top-k, top-p, as in Hugging Face generation. The random stream depends only on
// the seed, so a run is reproducible on any platform.
class Sampler {
 public:
  explicit Sampler(const SamplerConfig& config);

  std::int32_t sample(std::span<const float> logits);
  const SamplerConfig& config() const noexcept { return config_; }

 private:
  double next_uniform() noexcept;  // in [0, 1)

  SamplerConfig config_;
  std::uint64_t state_;
  std::vector<std::int32_t> candidates_;
  std::vector<double> probs_;
};

// Index of the largest logit; the lowest index wins ties.
std::int32_t argmax(std::span<const float> logits) noexcept;

}  // namespace kvb
