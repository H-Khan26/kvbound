// Beam search on the CPU. kvb::BeamSearch replays the traces of the Python reference
// (fixtures/tiny-whisper/beam_traces.safetensors, written by kvbound.fixture) and must take every
// decision the reference took; hand-built steps cover the edges; and the fixture model itself,
// run by the CPU executor of the Whisper programs, is searched to the reference's results.

#include "kvbound/beam_search.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <span>
#include <string>
#include <vector>

#include "kvbound/error.h"
#include "kvbound/json.h"
#include "kvbound/model_config.h"
#include "kvbound/program.h"
#include "kvbound/safetensors.h"
#include "kvbound/step.h"
#include "support/whisper_cpu.h"

namespace {

using kvb::Beam;
using kvb::BeamCandidate;
using kvb::BeamConfig;
using kvb::BeamHypothesis;
using kvb::BeamSearch;
using kvb::BeamStep;

using Tokens = std::vector<std::int32_t>;

const std::filesystem::path kFixture = KVB_WHISPER_FIXTURE_DIR;
const float kInf = std::numeric_limits<float>::infinity();
const BeamCandidate kPadding{.token = -1, .log_prob = -kInf};

// Scores agree to 1e-6, relative to the score above 1.
void expect_score(float actual, float expected) {
  EXPECT_NEAR(actual, expected, 1e-6 * std::max(1.0f, std::abs(expected)));
}

void expect_beam(const Beam& beam, int parent, std::int32_t token, float score) {
  EXPECT_EQ(beam.parent, parent);
  EXPECT_EQ(beam.token, token);
  expect_score(beam.score, score);
}

Tokens tokens_of(std::span<const std::int32_t> tokens) { return {tokens.begin(), tokens.end()}; }

// The BeamConfig of every trace in the file, by "<prompt>/<case>".
kvb::json::Value trace_configs(const kvb::SafetensorsFile& file) {
  return kvb::json::parse(file.metadata().at("configs"));
}

// One search of the reference, as kvbound.fixture.beam_trace stores it.
struct Trace {
  Trace(const kvb::SafetensorsFile& file, const std::string& name)
      : config(beam_config(trace_configs(file).at(name))),
        ids(ints(file, name, "ids")),
        candidate_ids(ints(file, name, "candidate_ids")),
        candidate_log_probs(floats(file, name, "candidate_log_probs")),
        beams(ints(file, name, "beams")),
        beam_scores(floats(file, name, "beam_scores")),
        finished(ints(file, name, "finished")),
        finished_scores(floats(file, name, "finished_scores")),
        hypotheses(ints(file, name, "hypotheses")),
        hypothesis_scores(floats(file, name, "hypothesis_scores")),
        n_steps(static_cast<std::size_t>(file.at(name + "/beams").shape.at(0))) {}

  static BeamConfig beam_config(const kvb::json::Value& v) {
    return {.num_beams = static_cast<int>(v.at("num_beams").as_int()),
            .eos_token_id = static_cast<std::int32_t>(v.at("eos_token_id").as_int()),
            .max_new_tokens = static_cast<int>(v.at("max_new_tokens").as_int()),
            .length_penalty = v.at("length_penalty").as_number()};
  }
  static std::span<const std::int32_t> ints(const kvb::SafetensorsFile& file,
                                            const std::string& name, const char* tensor) {
    return file.at(name + "/" + tensor).as<std::int32_t>();
  }
  static std::span<const float> floats(const kvb::SafetensorsFile& file, const std::string& name,
                                       const char* tensor) {
    return file.at(name + "/" + tensor).as<float>();
  }

  std::size_t num_beams() const { return static_cast<std::size_t>(config.num_beams); }
  std::size_t n_candidates() const { return candidate_ids.size() / (n_steps * num_beams()); }
  // The kept hypothesis `i` without its padding.
  Tokens hypothesis(std::size_t i) const {
    const std::size_t width = static_cast<std::size_t>(config.max_new_tokens);
    Tokens tokens;
    for (const std::int32_t token : hypotheses.subspan(i * width, width)) {
      if (token >= 0) tokens.push_back(token);
    }
    return tokens;
  }

  BeamConfig config;
  std::span<const std::int32_t> ids;
  std::span<const std::int32_t> candidate_ids;  // [steps, num_beams, n_candidates]
  std::span<const float> candidate_log_probs;
  std::span<const std::int32_t> beams;  // [steps, num_beams, (parent, token)]
  std::span<const float> beam_scores;
  std::span<const std::int32_t> finished;  // [n, (step, parent, token)]
  std::span<const float> finished_scores;
  std::span<const std::int32_t> hypotheses;  // [kept, max_new_tokens]
  std::span<const float> hypothesis_scores;
  std::size_t n_steps;
};

// "<prompt>/<case>" of every trace in the file.
std::vector<std::string> trace_names(const kvb::SafetensorsFile& file) {
  const kvb::json::Value configs = trace_configs(file);
  std::vector<std::string> names;
  for (const auto& [name, config] : configs.as_object()) names.push_back(name);
  return names;
}

TEST(BeamSearch, ReplaysTheReferenceTraces) {
  const kvb::SafetensorsFile file{kFixture / "beam_traces.safetensors"};
  const std::vector<std::string> names = trace_names(file);
  ASSERT_EQ(names.size(), 12u);
  // What the traces cover between them.
  bool stopped_early = false, ended_at_limit = false, dropped_a_hypothesis = false;
  for (const std::string& name : names) {
    SCOPED_TRACE(name);
    const Trace trace(file, name);
    const std::size_t n_beams = trace.num_beams(), k = trace.n_candidates();
    BeamSearch search(trace.config);
    ASSERT_EQ(static_cast<std::size_t>(search.n_candidates()), k);

    std::vector<BeamCandidate> candidates;
    std::size_t n_finished = 0;
    for (std::size_t s = 0; s < trace.n_steps; ++s) {
      SCOPED_TRACE("step " + std::to_string(s));
      ASSERT_FALSE(search.done());
      const std::size_t n_live = static_cast<std::size_t>(search.n_live());
      candidates.clear();
      for (std::size_t i = s * n_beams * k; i < (s * n_beams + n_live) * k; ++i) {
        candidates.push_back({trace.candidate_ids[i], trace.candidate_log_probs[i]});
      }
      const BeamStep& step = search.step(candidates);

      for (std::size_t b = 0; b < n_beams; ++b) {
        const std::size_t at = s * n_beams + b;
        if (b < step.beams.size()) {
          expect_beam(step.beams[b], trace.beams[2 * at], trace.beams[2 * at + 1],
                      trace.beam_scores[at]);
        } else {
          EXPECT_EQ(trace.beams[2 * at], -1) << "the reference has more live beams";
        }
      }
      for (const Beam& ended : step.finished) {
        ASSERT_LT(n_finished, trace.finished_scores.size());
        EXPECT_EQ(trace.finished[3 * n_finished], static_cast<std::int32_t>(s));
        expect_beam(ended, trace.finished[3 * n_finished + 1], trace.finished[3 * n_finished + 2],
                    trace.finished_scores[n_finished]);
        ++n_finished;
      }
      EXPECT_EQ(step.done, s + 1 == trace.n_steps);
      EXPECT_EQ(search.done(), step.done);
    }
    EXPECT_EQ(n_finished, trace.finished_scores.size());

    const std::span<const BeamHypothesis> kept = search.hypotheses();
    ASSERT_EQ(kept.size(), trace.hypothesis_scores.size());
    for (std::size_t i = 0; i < kept.size(); ++i) {
      EXPECT_EQ(kept[i].tokens, trace.hypothesis(i)) << "hypothesis " << i;
      expect_score(kept[i].score, trace.hypothesis_scores[i]);
    }
    // The best hypothesis is the end of the reference's whole sequence.
    const Tokens& best = search.best().tokens;
    ASSERT_LE(best.size(), trace.ids.size());
    EXPECT_EQ(best, tokens_of(trace.ids.last(best.size())));

    const int max_new_tokens = trace.config.max_new_tokens;
    stopped_early = stopped_early || static_cast<int>(trace.n_steps) < max_new_tokens;
    ended_at_limit = ended_at_limit || static_cast<int>(best.size()) == max_new_tokens;
    dropped_a_hypothesis = dropped_a_hypothesis || n_finished > kept.size();
  }
  EXPECT_TRUE(stopped_early);
  EXPECT_TRUE(ended_at_limit);
  EXPECT_TRUE(dropped_a_hypothesis);
}

// The candidates of the live beams of one step: the same number for each, beam after beam.
std::vector<BeamCandidate> beams(std::initializer_list<std::vector<BeamCandidate>> lists) {
  std::size_t width = 0;
  for (const auto& list : lists) width = std::max(width, list.size());
  std::vector<BeamCandidate> flat;
  for (const auto& list : lists) {
    flat.insert(flat.end(), list.begin(), list.end());
    flat.insert(flat.end(), width - list.size(), kPadding);
  }
  return flat;
}

TEST(BeamSearch, OneBeamIsGreedy) {
  BeamSearch search({.num_beams = 1, .eos_token_id = 9, .max_new_tokens = 4});
  EXPECT_EQ(search.n_live(), 1);
  EXPECT_EQ(search.n_candidates(), 2);
  // The end token in second place is not taken: the search follows the best token.
  const BeamStep& first = search.step(beams({{{3, -0.5f}, {9, -1.0f}}}));
  ASSERT_EQ(first.beams.size(), 1u);
  expect_beam(first.beams[0], 0, 3, -0.5f);
  EXPECT_TRUE(first.finished.empty());
  EXPECT_FALSE(first.done);
  EXPECT_EQ(tokens_of(search.tokens(0)), (Tokens{3}));
  // The end token in first place ends the search at once.
  const BeamStep& second = search.step(beams({{{4, -0.3f}, {9, -0.25f}}}));
  ASSERT_EQ(second.finished.size(), 1u);
  expect_beam(second.finished[0], 0, 9, -0.375f);  // (-0.5 - 0.25) / 2
  EXPECT_TRUE(second.done);
  EXPECT_EQ(search.best().tokens, (Tokens{3, 9}));
  expect_score(search.best().score, -0.375f);
  EXPECT_THROW(search.step(beams({{{4, -0.3f}}})), kvb::Error);
}

TEST(BeamSearch, TheLimitEndsEveryBeamWithoutAnEndToken) {
  BeamSearch search({.num_beams = 2, .eos_token_id = -1, .max_new_tokens = 2});
  const BeamStep& first = search.step(beams({{{1, -1.0f}, {2, -2.0f}, {3, -3.0f}, {4, -4.0f}}}));
  ASSERT_EQ(first.beams.size(), 2u);
  expect_beam(first.beams[0], 0, 1, -1.0f);
  expect_beam(first.beams[1], 0, 2, -2.0f);
  EXPECT_TRUE(first.finished.empty());
  EXPECT_THROW(search.best(), kvb::Error);  // nothing has finished yet

  // At the limit the best num_beams continuations become the hypotheses, whatever their tokens.
  const BeamStep& last = search.step(beams({{{5, -0.5f}, {6, -3.0f}}, {{7, -0.25f}, {8, -2.5f}}}));
  EXPECT_TRUE(last.done);
  EXPECT_TRUE(last.beams.empty());
  ASSERT_EQ(last.finished.size(), 2u);
  expect_beam(last.finished[0], 0, 5, -0.75f);   // (-1 - 0.5) / 2
  expect_beam(last.finished[1], 1, 7, -1.125f);  // (-2 - 0.25) / 2
  ASSERT_EQ(search.hypotheses().size(), 2u);
  EXPECT_EQ(search.hypotheses()[0].tokens, (Tokens{1, 5}));
  EXPECT_EQ(search.hypotheses()[1].tokens, (Tokens{2, 7}));
  EXPECT_EQ(search.n_live(), 0);
}

TEST(BeamSearch, StopsWhenEveryBeamEndsEarly) {
  BeamSearch search({.num_beams = 2, .eos_token_id = 0, .max_new_tokens = 10});
  search.step(beams({{{1, -0.5f}, {2, -1.0f}, {3, -4.0f}, {4, -5.0f}}}));
  // Both beams put the end token first. The beams that go on score far below the two
  // hypotheses, so the search is over after 2 of 10 tokens.
  const BeamStep& step = search.step(beams({{{0, -0.25f}, {5, -4.0f}}, {{0, -0.5f}, {6, -6.0f}}}));
  ASSERT_EQ(step.finished.size(), 2u);
  expect_beam(step.finished[0], 0, 0, -0.375f);
  expect_beam(step.finished[1], 1, 0, -0.75f);
  ASSERT_EQ(step.beams.size(), 2u);
  expect_beam(step.beams[0], 0, 5, -4.5f);
  EXPECT_TRUE(step.done);
  EXPECT_EQ(search.best().tokens, (Tokens{1, 0}));
}

TEST(BeamSearch, GoesOnWhileALiveBeamCanBeatAKeptHypothesis) {
  BeamSearch search({.num_beams = 2, .eos_token_id = 0, .max_new_tokens = 10});
  // The end token is the best first token: a hypothesis of one token, and the search goes on
  // because one hypothesis is fewer than num_beams.
  const BeamStep& first = search.step(beams({{{0, -1.0f}, {1, -1.5f}, {2, -2.0f}}}));
  ASSERT_EQ(first.finished.size(), 1u);
  expect_beam(first.finished[0], 0, 0, -1.0f);
  ASSERT_EQ(first.beams.size(), 2u);
  EXPECT_FALSE(first.done);

  // Ranked: (beam 0, 3) -1.625, (0, end) -1.75, (1, end) -2.5, (1, 4) -5. The second end ranks
  // below num_beams: it is neither a hypothesis nor a beam.
  const BeamStep& second =
      search.step(beams({{{3, -0.125f}, {0, -0.25f}}, {{0, -0.5f}, {4, -3.0f}}}));
  ASSERT_EQ(second.finished.size(), 1u);
  expect_beam(second.finished[0], 0, 0, -0.875f);
  ASSERT_EQ(second.beams.size(), 2u);
  expect_beam(second.beams[0], 0, 3, -1.625f);
  expect_beam(second.beams[1], 1, 4, -5.0f);
  // Two hypotheses are kept, and the best beam at -1.625 / 2 still beats the worse one at -1.
  EXPECT_FALSE(second.done);
  EXPECT_EQ(search.hypotheses()[0].tokens, (Tokens{1, 0}));
  EXPECT_EQ(search.hypotheses()[1].tokens, (Tokens{0}));

  // A third hypothesis at -2.125 / 3 crowds out the worst. The best beam now scores
  // -2.625 / 3 = -0.875, which equals the worst kept hypothesis and does not beat it.
  const BeamStep& third = search.step(beams({{{0, -0.5f}, {5, -1.0f}}, {{6, -1.0f}, {0, -2.0f}}}));
  ASSERT_EQ(third.finished.size(), 1u);
  expect_score(third.finished[0].score, -2.125f / 3.0f);
  expect_beam(third.beams[0], 0, 5, -2.625f);
  EXPECT_TRUE(third.done);
  ASSERT_EQ(search.hypotheses().size(), 2u);
  EXPECT_EQ(search.hypotheses()[0].tokens, (Tokens{1, 3, 0}));
  EXPECT_EQ(search.hypotheses()[1].tokens, (Tokens{1, 0}));
}

TEST(BeamSearch, BreaksTiesByBeamThenLogProbThenToken) {
  BeamSearch search({.num_beams = 2, .eos_token_id = -1, .max_new_tokens = 3});
  // Equal log-probabilities in one beam: the lower token leads.
  const BeamStep& first = search.step(beams({{{7, -1.0f}, {3, -1.0f}, {5, -1.0f}, {9, -2.0f}}}));
  expect_beam(first.beams[0], 0, 3, -1.0f);
  expect_beam(first.beams[1], 0, 5, -1.0f);

  // Equal scores in two beams: the lower beam leads. The second beam's first token is so
  // unlikely that FP32 cannot tell its continuations apart once it is added.
  const float huge = -16777216.0f;  // -2^24: adding -0.25 or -0.5 rounds back to it
  const BeamStep& second = search.step(beams({{{1, -0.5f}, {2, huge}}, {{1, -0.5f}, {4, -0.75f}}}));
  expect_beam(second.beams[0], 0, 1, -1.5f);
  expect_beam(second.beams[1], 1, 1, -1.5f);

  // Beam 0 offers three tokens whose scores are equal after rounding: the larger
  // log-probability leads, then the lower token.
  BeamSearch rounded({.num_beams = 2, .eos_token_id = -1, .max_new_tokens = 3});
  rounded.step(beams({{{1, huge}}}));
  const BeamStep& tied = rounded.step(beams({{{6, -0.5f}, {8, -0.25f}, {2, -0.5f}}}));
  ASSERT_EQ(tied.beams.size(), 2u);
  expect_beam(tied.beams[0], 0, 8, huge);
  expect_beam(tied.beams[1], 0, 2, huge);
}

TEST(BeamSearch, AnEarlierHypothesisLeadsATie) {
  BeamSearch search({.num_beams = 2, .eos_token_id = 0, .max_new_tokens = 5});
  search.step(beams({{{0, -1.0f}, {1, -0.5f}, {2, -1.25f}}}));
  // A second hypothesis with the same score, -2 / 2, goes behind the first.
  search.step(beams({{{0, -1.5f}, {4, -0.25f}}, {{5, -2.0f}, {6, -3.0f}}}));
  ASSERT_EQ(search.hypotheses().size(), 2u);
  EXPECT_EQ(search.hypotheses()[0].tokens, (Tokens{0}));
  EXPECT_EQ(search.hypotheses()[1].tokens, (Tokens{1, 0}));
  EXPECT_EQ(search.hypotheses()[0].score, search.hypotheses()[1].score);
  // A third, -3 / 3, finds the kept set full of its equals and is dropped.
  const BeamStep& third =
      search.step(beams({{{0, -2.25f}, {7, -0.25f}}, {{8, -5.0f}, {9, -6.0f}}}));
  ASSERT_EQ(third.finished.size(), 1u);
  expect_score(third.finished[0].score, -1.0f);
  EXPECT_EQ(search.hypotheses()[0].tokens, (Tokens{0}));
  EXPECT_EQ(search.hypotheses()[1].tokens, (Tokens{1, 0}));
  EXPECT_FALSE(third.done);
}

TEST(BeamSearch, AppliesTheLengthPenalty) {
  // Two hypotheses of lengths 1 and 2 with log-probability sums -1 and -1.5.
  const auto best = [](double length_penalty) {
    BeamSearch search(
        {.num_beams = 2, .eos_token_id = 0, .max_new_tokens = 2, .length_penalty = length_penalty});
    search.step(beams({{{0, -1.0f}, {1, -0.5f}, {2, -3.0f}}}));
    search.step(beams({{{0, -1.0f}, {3, -9.0f}}, {{4, -9.0f}, {5, -9.5f}}}));
    return search.best();
  };
  // Penalty 0 compares the sums; 1 divides by the length; 2 by its square; 0.5 by its root,
  // which leaves the shorter one ahead.
  EXPECT_EQ(best(0.0).tokens, (Tokens{0}));
  expect_score(best(0.0).score, -1.0f);
  EXPECT_EQ(best(1.0).tokens, (Tokens{1, 0}));
  expect_score(best(1.0).score, -0.75f);
  EXPECT_EQ(best(2.0).tokens, (Tokens{1, 0}));
  expect_score(best(2.0).score, -0.375f);
  EXPECT_EQ(best(0.5).tokens, (Tokens{0}));
  expect_score(best(0.5).score, -1.0f);
}

TEST(BeamSearch, IgnoresPaddingAndRunsWithFewerBeams) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  BeamSearch search({.num_beams = 3, .eos_token_id = 0, .max_new_tokens = 8});
  // Two finite candidates: two live beams, not three.
  const BeamStep& first =
      search.step(beams({{{1, -1.0f}, kPadding, {2, -2.0f}, {3, -kInf}, {4, nan}, kPadding}}));
  ASSERT_EQ(first.beams.size(), 2u);
  EXPECT_EQ(search.n_live(), 2);
  // A beam with nothing to offer has no continuation.
  const BeamStep& second = search.step(beams({{kPadding, kPadding}, {{5, -1.0f}, kPadding}}));
  ASSERT_EQ(second.beams.size(), 1u);
  expect_beam(second.beams[0], 1, 5, -3.0f);
  EXPECT_EQ(tokens_of(search.tokens(0)), (Tokens{2, 5}));
  // With no candidate at all the search is over, without a result.
  const BeamStep& third = search.step({});
  EXPECT_TRUE(third.done);
  EXPECT_TRUE(third.beams.empty());
  EXPECT_THROW(search.best(), kvb::Error);
}

TEST(BeamSearch, RejectsMisuse) {
  EXPECT_THROW(BeamSearch({.num_beams = 0, .max_new_tokens = 4}), kvb::Error);
  EXPECT_THROW(BeamSearch({.num_beams = 2, .max_new_tokens = 0}), kvb::Error);
  BeamSearch search({.num_beams = 2, .eos_token_id = 0, .max_new_tokens = 4});
  search.step(beams({{{1, -1.0f}, {2, -2.0f}}}));
  // Three entries do not divide among two live beams.
  const std::vector<BeamCandidate> odd{{1, -1.0f}, {2, -1.0f}, {3, -1.0f}};
  EXPECT_THROW(search.step(odd), kvb::Error);
  EXPECT_THROW(search.tokens(2), kvb::Error);
  EXPECT_THROW(search.tokens(-1), kvb::Error);
}

// A stand-in model: the candidates of a beam depend on its tokens alone. Log-probabilities are
// multiples of 1/4, so equal scores are common within a beam and between beams.
std::vector<BeamCandidate> table_candidates(std::span<const std::int32_t> tokens, int vocab,
                                            std::uint32_t seed) {
  std::uint32_t state = seed;
  for (const std::int32_t token : tokens)
    state = (state ^ static_cast<std::uint32_t>(token)) * 16777619u;
  std::vector<BeamCandidate> candidates;
  for (std::int32_t token = 0; token < vocab; ++token) {
    state = state * 1664525u + 1013904223u;
    candidates.push_back({token, -0.25f * static_cast<float>(1 + (state >> 16) % 12)});
  }
  return candidates;
}

// Each beam offering its n_candidates() = num_beams + 1 best candidates gives the search the
// whole vocabulary would: every step takes the same decisions.
TEST(BeamSearch, TheBestCandidatesOfEachBeamAreEnough) {
  const int vocab = 24;
  int compared = 0;
  for (std::uint32_t seed = 1; seed <= 40; ++seed) {
    const BeamConfig config{.num_beams = 1 + static_cast<int>(seed % 5),
                            .eos_token_id = 0,
                            .max_new_tokens = 8,
                            .length_penalty = seed % 3 == 0 ? 0.6 : 1.0};
    SCOPED_TRACE("seed " + std::to_string(seed));
    BeamSearch whole(config), top(config);
    while (!whole.done()) {
      ASSERT_FALSE(top.done());
      ASSERT_EQ(whole.n_live(), top.n_live());
      std::vector<BeamCandidate> all, best;
      for (int b = 0; b < whole.n_live(); ++b) {
        ASSERT_EQ(tokens_of(whole.tokens(b)), tokens_of(top.tokens(b)));
        std::vector<BeamCandidate> candidates = table_candidates(whole.tokens(b), vocab, seed);
        all.insert(all.end(), candidates.begin(), candidates.end());
        std::stable_sort(
            candidates.begin(), candidates.end(),
            [](const BeamCandidate& x, const BeamCandidate& y) { return x.log_prob > y.log_prob; });
        best.insert(best.end(), candidates.begin(), candidates.begin() + top.n_candidates());
      }
      const BeamStep expected = whole.step(all);
      const BeamStep& actual = top.step(best);
      ASSERT_EQ(actual.beams.size(), expected.beams.size());
      for (std::size_t i = 0; i < expected.beams.size(); ++i) {
        const Beam& beam = expected.beams[i];
        expect_beam(actual.beams[i], beam.parent, beam.token, beam.score);
        EXPECT_EQ(actual.beams[i].score, beam.score);
      }
      ASSERT_EQ(actual.finished.size(), expected.finished.size());
      for (std::size_t i = 0; i < expected.finished.size(); ++i) {
        const Beam& ended = expected.finished[i];
        expect_beam(actual.finished[i], ended.parent, ended.token, ended.score);
      }
      EXPECT_EQ(actual.done, expected.done);
      ++compared;
    }
    EXPECT_TRUE(top.done());
    EXPECT_EQ(top.best().tokens, whole.best().tokens);
    EXPECT_EQ(top.best().score, whole.best().score);
  }
  EXPECT_GT(compared, 100);
}

TEST(BeamCandidates, AreTheLargestLogProbsOfTheWholeRow) {
  // Probabilities 1/8, 2/8, 4/8, 1/8.
  const std::vector<float> logits{std::log(1.0f), std::log(2.0f), std::log(4.0f), std::log(1.0f)};
  std::vector<BeamCandidate> out{{42, -1.0f}};
  kvb::append_beam_candidates(logits, {}, 3, out);
  ASSERT_EQ(out.size(), 4u);
  EXPECT_EQ(out[0].token, 42);  // what was there stays
  EXPECT_EQ(out[1].token, 2);
  EXPECT_NEAR(out[1].log_prob, std::log(0.5), 1e-6);
  EXPECT_EQ(out[2].token, 1);
  EXPECT_NEAR(out[2].log_prob, std::log(0.25), 1e-6);
  EXPECT_EQ(out[3].token, 0);  // the lower of two equal tokens
  EXPECT_NEAR(out[3].log_prob, std::log(0.125), 1e-6);

  // A suppressed token is left out, but the others keep the probabilities they had with it.
  // Fewer finite candidates than k leaves padding.
  out.clear();
  const std::vector<std::int32_t> suppressed{2};
  kvb::append_beam_candidates(logits, suppressed, 5, out);
  ASSERT_EQ(out.size(), 5u);
  EXPECT_EQ(out[0].token, 1);
  EXPECT_NEAR(out[0].log_prob, std::log(0.25), 1e-6);
  EXPECT_EQ(out[1].token, 0);
  EXPECT_EQ(out[2].token, 3);
  EXPECT_NEAR(out[2].log_prob, std::log(0.125), 1e-6);
  EXPECT_FALSE(std::isfinite(out[3].log_prob));
  EXPECT_FALSE(std::isfinite(out[4].log_prob));

  // A logit of -inf has probability 0 and is no candidate.
  out.clear();
  kvb::append_beam_candidates(std::vector<float>{0.0f, -kInf, 0.0f}, {}, 3, out);
  EXPECT_EQ(out[0].token, 0);
  EXPECT_EQ(out[1].token, 2);
  EXPECT_NEAR(out[1].log_prob, std::log(0.5), 1e-6);
  EXPECT_FALSE(std::isfinite(out[2].log_prob));

  const std::vector<std::int32_t> outside{4};
  EXPECT_THROW(kvb::append_beam_candidates(logits, outside, 2, out), kvb::Error);
  EXPECT_THROW(kvb::append_beam_candidates(logits, {}, 0, out), kvb::Error);
  EXPECT_THROW(kvb::append_beam_candidates({}, {}, 2, out), kvb::Error);
}

// The fixture model as the CPU executor runs it: one sequence row, whose cache every beam
// overwrites by running its whole prefix again.
struct FixtureModel {
  explicit FixtureModel(const std::string& prompt) {
    // The file stores features [n_mels, frames]; the encoder takes one row per frame.
    const auto stored = goldens.at(prompt + "/input_features").as<float>();
    const std::size_t n_mels = static_cast<std::size_t>(config.n_mels);
    const std::size_t n_frames = static_cast<std::size_t>(config.n_frames());
    std::vector<float> mel(stored.size());
    for (std::size_t m = 0; m < n_mels; ++m) {
      for (std::size_t f = 0; f < n_frames; ++f) mel[f * n_mels + m] = stored[m * n_frames + f];
    }
    std::vector<kvb::StepRow> rows;
    const std::int32_t n_ctx = static_cast<std::int32_t>(config.max_source_positions);
    for (std::int32_t p = 0; p < n_ctx; ++p) rows.push_back({0, p, 0, p});
    model.encode(rows, mel);
  }

  // Logits [vocab] of the token after `ids`, before any suppression.
  std::vector<float> next_logits(std::span<const std::int32_t> ids) {
    std::vector<kvb::StepRow> rows;
    for (std::int32_t p = 0; p < static_cast<std::int32_t>(ids.size()); ++p) {
      rows.push_back({ids[p], p, 0, p});
    }
    const std::vector<std::int32_t> last{static_cast<std::int32_t>(ids.size()) - 1};
    kvb::test::Capture capture;
    model.decode(rows, last, &capture);
    return capture.at("logits");
  }

  kvb::WhisperConfig config = kvb::load_whisper_config(kFixture / "config.json");
  kvb::WhisperRules rules = kvb::load_whisper_rules(kFixture / "generation.json");
  kvb::WhisperPrograms programs = kvb::build_whisper_programs(config, rules, false);
  kvb::SafetensorsFile weights{kFixture / "model.safetensors"};
  kvb::SafetensorsFile goldens{kFixture / "goldens.safetensors"};
  kvb::test::CpuWhisper model{config, programs, weights, 1};
};

class BeamOnFixture : public testing::TestWithParam<const char*> {};

// The candidate interface carries a real model: logits of the CPU executor, turned into
// candidates and searched, give the sequence and score of the reference.
TEST_P(BeamOnFixture, FindsTheReferenceResult) {
  const std::string name = GetParam();
  const kvb::SafetensorsFile file{kFixture / "beam_traces.safetensors"};
  const Trace trace(file, name);
  FixtureModel fx(name.substr(0, name.find('/')));
  const Tokens& prompt = fx.rules.prompt;
  Tokens at_begin = fx.rules.suppress_tokens;
  at_begin.insert(at_begin.end(), fx.rules.begin_suppress_tokens.begin(),
                  fx.rules.begin_suppress_tokens.end());

  BeamSearch search(trace.config);
  std::vector<BeamCandidate> candidates;
  std::size_t n_steps = 0;
  for (; !search.done(); ++n_steps) {
    candidates.clear();
    for (int b = 0; b < search.n_live(); ++b) {
      Tokens ids = prompt;
      ids.insert(ids.end(), search.tokens(b).begin(), search.tokens(b).end());
      const bool begin = ids.size() == prompt.size();
      kvb::append_beam_candidates(fx.next_logits(ids), begin ? at_begin : fx.rules.suppress_tokens,
                                  search.n_candidates(), candidates);
    }
    search.step(candidates);
  }
  EXPECT_EQ(n_steps, trace.n_steps);
  Tokens ids = prompt;
  ids.insert(ids.end(), search.best().tokens.begin(), search.best().tokens.end());
  EXPECT_EQ(ids, tokens_of(trace.ids));
  // The executor's scores are within 4e-7 of the reference's. The closest decision on these
  // cases, p0/beam5's best hypothesis over the second, is 1.2e-5.
  EXPECT_NEAR(search.best().score, trace.hypothesis_scores[0], 1e-5f);
}

INSTANTIATE_TEST_SUITE_P(BeamSearch, BeamOnFixture,
                         testing::Values("p0/beam5", "p1/beam5", "p2/beam5", "p1/beam1",
                                         "p1/beam5.eos38", "p0/beam5.eos72.penalty0.6",
                                         "p2/beam5.eos91.penalty2"));

}  // namespace
