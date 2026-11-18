#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace kvb {

// Beam search as Hugging Face `generate(num_beams=N, do_sample=False)` runs it, as a state
// machine on the CPU. It knows no model: each step it takes the best next tokens of every live
// beam and answers with the beams of the next step. python/kvbound/ref/beam.py is the same
// machine and states the rules; tests replay its traces against this one.
//
// The caller's loop, one request:
//   BeamSearch search(config);  // one live beam, holding the prompt
//   while (!search.done()) {
//     // Run the model on every live beam b in [0, n_live()): the whole prompt in the first
//     // step, the beam's last token after that. Row b of the logits gives beam b's candidates;
//     // `suppressed` is the suppress list, plus the begin-suppress list in the first step.
//     for each b: append_beam_candidates(logits_b, suppressed, search.n_candidates(), candidates);
//     const BeamStep& step = search.step(candidates);
//     // Live beam i of the next step continues old beam step.beams[i].parent with token
//     // step.beams[i].token: fork the parents' cache blocks in the order block_allocator.h
//     // gives, then feed each beam its token at the position after the cached ones.
//   }
//   search.best();  // generated tokens, the end token included, and the final score
//
// Hugging Face starts beams 1..num_beams-1 at -1e9 so that the first step draws from beam 0
// alone; one live beam in the first step is the same search with the prompt run once. Later
// steps have num_beams live beams unless a beam offers fewer finite candidates than
// n_candidates() (Hugging Face would then continue beams scored near -1e9; none of its usual
// processors leave so few). At max_new_tokens every continuation ends and the best num_beams of
// the step become hypotheses. Scores are FP32 and use the same operations as the reference, so
// both take the same decisions from the same candidates.

struct BeamConfig {
  int num_beams = 1;
  std::int32_t eos_token_id = -1;  // a beam ends on this token; -1 ends beams by length alone
  int max_new_tokens = 0;          // a beam ends when it holds this many tokens
  // A hypothesis scores sum of log-probabilities / length ^ length_penalty, where length counts
  // its generated tokens, the end-of-sequence token included.
  double length_penalty = 1.0;
};

struct BeamCandidate {
  std::int32_t token = 0;
  float log_prob = 0.0f;  // not finite: the entry is padding and is ignored
};

// A continuation: the tokens of beam `parent` of the step before, then `token`.
struct Beam {
  int parent = 0;
  std::int32_t token = 0;
  float score = 0.0f;  // sum of log-probabilities when live, the final score when finished
};

struct BeamStep {
  std::vector<Beam> beams;  // live beams of the next step, best first; unused once done
  // Continuations that ended in this step, best first. Not all need be kept as hypotheses.
  std::vector<Beam> finished;
  bool done = false;
};

struct BeamHypothesis {
  std::vector<std::int32_t> tokens;  // generated tokens, the end-of-sequence token included
  float score = 0.0f;
};

class BeamSearch {
 public:
  // Throws kvb::Error unless num_beams and max_new_tokens are above 0.
  explicit BeamSearch(const BeamConfig& config);

  const BeamConfig& config() const noexcept { return config_; }
  bool done() const noexcept { return done_; }
  // Beams the next step() takes candidates for.
  int n_live() const noexcept { return static_cast<int>(scores_.size()); }
  // Candidates per beam that give every step the decisions the whole vocabulary would:
  // num_beams + 1. A step ranks continuations by (score descending, beam ascending,
  // log-probability descending, token ascending), keeps the best num_beams that go on, and makes
  // hypotheses of those that end among the best num_beams overall. Adding a beam's score to two
  // log-probabilities in FP32 keeps their order or makes them equal, so within one beam the
  // ranking is (log-probability descending, token ascending). A candidate behind its beam's best
  // num_beams + 1 thus has num_beams + 1 continuations ahead of it, at most one of them the end
  // token: it neither ends among the best num_beams nor is among the best num_beams that go on.
  // (Hugging Face ranks the best 2 * num_beams of all beams, which hold every continuation
  // these decisions use.)
  int n_candidates() const noexcept { return config_.num_beams + 1; }
  // Generated tokens of a live beam; valid until the next step().
  std::span<const std::int32_t> tokens(int beam) const;

  // Advances every live beam by one token. `candidates` holds the same number of entries for
  // each live beam, beam after beam, in any order within a beam. A beam's entries must include
  // its n_candidates() best by (log-probability descending, token ascending), or every token
  // with a finite log-probability when there are fewer. Log-probabilities come from a
  // log-softmax over the whole vocabulary; a token that must not be generated is left out or
  // given -inf after it. The result stays valid until the next call. Throws kvb::Error when the
  // search is done or the entries do not divide among the live beams.
  const BeamStep& step(std::span<const BeamCandidate> candidates);

  // Kept hypotheses, best first: at most num_beams. On a tie the one that finished first leads.
  std::span<const BeamHypothesis> hypotheses() const noexcept { return hypotheses_; }
  // The result once done(). Throws kvb::Error when no hypothesis has finished.
  const BeamHypothesis& best() const;

 private:
  struct Ranked {
    float score;
    int beam;
    BeamCandidate candidate;
  };

  BeamConfig config_;
  bool done_ = false;
  int n_generated_ = 0;
  // Live beams, parallel: generated tokens and sum of log-probabilities.
  std::vector<std::vector<std::int32_t>> tokens_;
  std::vector<float> scores_;
  std::vector<BeamHypothesis> hypotheses_;
  BeamStep step_;
  std::vector<Ranked> ranked_;
};

// Appends to `out` exactly k candidates of one beam from its logits row [vocab]: the k largest
// log-probabilities, lower token first on a tie, padded when fewer are finite. The log-softmax
// runs over the whole row, `suppressed` tokens included, and those tokens are then left out,
// which is the order Hugging Face applies them in. The logits must therefore be the ones before
// any suppression: a row with -inf already written into it gives other log-probabilities.
void append_beam_candidates(std::span<const float> logits, std::span<const std::int32_t> suppressed,
                            int k, std::vector<BeamCandidate>& out);

}  // namespace kvb
