#include "kvbound/sampler.h"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "kvbound/error.h"

namespace kvb {

std::int32_t argmax(std::span<const float> logits) noexcept {
  return static_cast<std::int32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
}

Sampler::Sampler(const SamplerConfig& config) : config_(config), state_(config.seed) {
  if (config.top_k < 0 || !(config.top_p > 0.0f) || config.top_p > 1.0f) {
    throw Error("sampler needs top_k >= 0 and 0 < top_p <= 1");
  }
}

// SplitMix64: a full-period 64-bit generator with a one-word state.
double Sampler::next_uniform() noexcept {
  std::uint64_t z = (state_ += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  z ^= z >> 31;
  return static_cast<double>(z >> 11) * 0x1.0p-53;
}

std::int32_t Sampler::sample(std::span<const float> logits) {
  if (logits.empty()) throw Error("cannot sample from empty logits");
  if (config_.greedy()) return argmax(logits);

  const auto by_logit = [&](std::int32_t a, std::int32_t b) {
    return logits[a] > logits[b] || (logits[a] == logits[b] && a < b);
  };
  candidates_.resize(logits.size());
  std::iota(candidates_.begin(), candidates_.end(), 0);
  if (config_.top_k > 0 && static_cast<std::size_t>(config_.top_k) < candidates_.size()) {
    std::nth_element(candidates_.begin(), candidates_.begin() + config_.top_k - 1,
                     candidates_.end(), by_logit);
    candidates_.resize(config_.top_k);
  }
  std::sort(candidates_.begin(), candidates_.end(), by_logit);

  // Softmax over the candidates, largest first. Subtracting the top logit keeps exp() in range.
  const double top = logits[candidates_.front()];
  probs_.resize(candidates_.size());
  double total = 0.0;
  for (std::size_t i = 0; i < candidates_.size(); ++i) {
    probs_[i] = std::exp((logits[candidates_[i]] - top) / config_.temperature);
    total += probs_[i];
  }

  // Nucleus: the shortest prefix whose probability reaches top_p.
  std::size_t kept = candidates_.size();
  if (config_.top_p < 1.0f) {
    double cumulative = 0.0;
    for (std::size_t i = 0; i < candidates_.size(); ++i) {
      cumulative += probs_[i];
      if (cumulative >= config_.top_p * total) {
        kept = i + 1;
        total = cumulative;
        break;
      }
    }
  }

  double target = next_uniform() * total;
  for (std::size_t i = 0; i + 1 < kept; ++i) {
    target -= probs_[i];
    if (target < 0.0) return candidates_[i];
  }
  return candidates_[kept - 1];
}

}  // namespace kvb
