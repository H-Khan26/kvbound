#include "kvbound/beam_search.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>

#include "kvbound/error.h"

namespace kvb {

namespace {

// The divisor of a hypothesis of `length` tokens. Hugging Face computes the power in FP64 and
// divides in FP32.
float length_divisor(int length, double length_penalty) {
  return static_cast<float>(std::pow(static_cast<double>(length), length_penalty));
}

}  // namespace

BeamSearch::BeamSearch(const BeamConfig& config) : config_(config), tokens_(1), scores_{0.0f} {
  if (config.num_beams <= 0 || config.max_new_tokens <= 0) {
    throw Error("beam search needs num_beams > 0 and max_new_tokens > 0");
  }
}

std::span<const std::int32_t> BeamSearch::tokens(int beam) const {
  if (beam < 0 || beam >= n_live()) throw Error("no live beam " + std::to_string(beam));
  return tokens_[beam];
}

const BeamHypothesis& BeamSearch::best() const {
  if (hypotheses_.empty()) throw Error("no beam has finished");
  return hypotheses_.front();
}

const BeamStep& BeamSearch::step(std::span<const BeamCandidate> candidates) {
  const std::size_t n_live = scores_.size();
  if (done_) throw Error("the beam search is done");
  if (candidates.size() % n_live != 0) {
    throw Error("beam candidates do not divide among the live beams");
  }
  const std::size_t per_beam = candidates.size() / n_live;
  const std::size_t num_beams = static_cast<std::size_t>(config_.num_beams);

  ranked_.clear();
  for (std::size_t beam = 0; beam < n_live; ++beam) {
    for (const BeamCandidate& c : candidates.subspan(beam * per_beam, per_beam)) {
      if (!std::isfinite(c.log_prob)) continue;
      ranked_.push_back({scores_[beam] + c.log_prob, static_cast<int>(beam), c});
    }
  }
  const auto ahead = [](const Ranked& a, const Ranked& b) {
    if (a.score != b.score) return a.score > b.score;
    if (a.beam != b.beam) return a.beam < b.beam;
    if (a.candidate.log_prob != b.candidate.log_prob) {
      return a.candidate.log_prob > b.candidate.log_prob;
    }
    return a.candidate.token < b.candidate.token;
  };
  const std::size_t kept = std::min(ranked_.size(), 2 * num_beams);
  std::partial_sort(ranked_.begin(), ranked_.begin() + static_cast<std::ptrdiff_t>(kept),
                    ranked_.end(), ahead);
  ranked_.resize(kept);

  // Every continuation of this step has the same length.
  const int length = n_generated_ + 1;
  const bool at_limit = length == config_.max_new_tokens;
  const float divisor = length_divisor(length, config_.length_penalty);
  step_.beams.clear();
  step_.finished.clear();
  for (std::size_t rank = 0; rank < ranked_.size(); ++rank) {
    const Ranked& r = ranked_[rank];
    if (r.candidate.token == config_.eos_token_id || at_limit) {
      // An ended continuation below the best num_beams is dropped, not kept as a hypothesis.
      if (rank < num_beams) {
        step_.finished.push_back({r.beam, r.candidate.token, r.score / divisor});
      }
    } else if (step_.beams.size() < num_beams) {
      step_.beams.push_back({r.beam, r.candidate.token, r.score});
    }
  }

  for (const Beam& ended : step_.finished) {
    hypotheses_.push_back({.tokens = tokens_[ended.parent], .score = ended.score});
    hypotheses_.back().tokens.push_back(ended.token);
  }
  // Stable, so on a tie the hypothesis kept longest stays ahead.
  std::stable_sort(
      hypotheses_.begin(), hypotheses_.end(),
      [](const BeamHypothesis& a, const BeamHypothesis& b) { return a.score > b.score; });
  if (hypotheses_.size() > num_beams) hypotheses_.resize(num_beams);

  std::vector<std::vector<std::int32_t>> next;
  next.reserve(step_.beams.size());
  scores_.clear();
  for (const Beam& beam : step_.beams) {
    next.push_back(tokens_[beam.parent]);
    next.back().push_back(beam.token);
    scores_.push_back(beam.score);
  }
  tokens_ = std::move(next);
  n_generated_ = length;

  if (!scores_.empty() && hypotheses_.size() == num_beams) {
    // The best live beam, scored as if it ended at this length, must beat the worst kept
    // hypothesis for the search to go on.
    done_ = !(scores_.front() / divisor > hypotheses_.back().score);
  } else {
    done_ = scores_.empty();
  }
  step_.done = done_;
  return step_;
}

void append_beam_candidates(std::span<const float> logits, std::span<const std::int32_t> suppressed,
                            int k, std::vector<BeamCandidate>& out) {
  if (logits.empty() || k <= 0) throw Error("beam candidates need logits and k > 0");
  const float inf = std::numeric_limits<float>::infinity();
  // log_softmax with the sum in FP64: log p = x - max - log(sum(exp(x - max))).
  const double top = *std::max_element(logits.begin(), logits.end());
  double total = 0.0;
  for (const float x : logits) total += std::exp(x - top);
  const double log_total = top + std::log(total);

  std::vector<bool> skip(logits.size(), false);
  for (const std::int32_t token : suppressed) {
    if (token < 0 || static_cast<std::size_t>(token) >= logits.size()) {
      throw Error("suppressed token " + std::to_string(token) + " is outside the vocabulary");
    }
    skip[token] = true;
  }

  // The k best so far, best first. Tokens arrive in ascending order and a later one must be
  // strictly better to pass an earlier one, so the lower token leads a tie.
  const std::size_t first = out.size();
  for (std::size_t token = 0; token < logits.size(); ++token) {
    if (skip[token]) continue;
    const float log_prob = static_cast<float>(logits[token] - log_total);
    if (!std::isfinite(log_prob)) continue;
    const std::size_t n = out.size() - first;
    if (n == static_cast<std::size_t>(k) && !(log_prob > out.back().log_prob)) continue;
    if (n == static_cast<std::size_t>(k)) out.pop_back();
    const auto at =
        std::upper_bound(out.begin() + static_cast<std::ptrdiff_t>(first), out.end(), log_prob,
                         [](float value, const BeamCandidate& c) { return value > c.log_prob; });
    out.insert(at, {static_cast<std::int32_t>(token), log_prob});
  }
  out.resize(first + static_cast<std::size_t>(k), BeamCandidate{.token = -1, .log_prob = -inf});
}

}  // namespace kvb
