#include "kvbound/sampler.h"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "kvbound/error.h"

namespace {

using kvb::Sampler;
using kvb::SamplerConfig;

// log of the probabilities 0.5, 0.25, 0.125, 0.0625, 0.0625, shuffled.
const std::vector<float> kLogits = {std::log(0.125f), std::log(0.5f), std::log(0.0625f),
                                    std::log(0.25f), std::log(0.0625f)};

std::vector<double> frequencies(const SamplerConfig& config, int draws) {
  Sampler sampler(config);
  std::vector<double> counts(kLogits.size(), 0.0);
  for (int i = 0; i < draws; ++i) counts[sampler.sample(kLogits)] += 1.0 / draws;
  return counts;
}

TEST(Sampler, GreedyTakesTheLargestLogitAndTheFirstOfATie) {
  Sampler sampler(SamplerConfig{});
  EXPECT_EQ(sampler.sample(kLogits), 1);
  EXPECT_EQ(sampler.sample(std::vector<float>{1.0f, 3.0f, 3.0f, 2.0f}), 1);
  EXPECT_EQ(kvb::argmax(std::vector<float>{-2.0f, -2.0f}), 0);
}

TEST(Sampler, TemperatureOneDrawsFromTheSoftmax) {
  const std::vector<double> freq = frequencies({.temperature = 1.0f, .seed = 1}, 40000);
  const double expected[] = {0.125, 0.5, 0.0625, 0.25, 0.0625};
  for (std::size_t i = 0; i < freq.size(); ++i) EXPECT_NEAR(freq[i], expected[i], 0.01) << i;
}

TEST(Sampler, TemperatureSharpensTheDistribution) {
  // At temperature 0.5 probabilities are squared and renormalized: 0.25 / 0.3359375 for the top.
  const std::vector<double> freq = frequencies({.temperature = 0.5f, .seed = 2}, 40000);
  EXPECT_NEAR(freq[1], 0.25 / 0.3359375, 0.01);
}

TEST(Sampler, TopKKeepsOnlyTheLargest) {
  const std::vector<double> freq = frequencies({.temperature = 1.0f, .top_k = 2, .seed = 3}, 40000);
  EXPECT_NEAR(freq[1], 0.5 / 0.75, 0.01);
  EXPECT_NEAR(freq[3], 0.25 / 0.75, 0.01);
  EXPECT_EQ(freq[0] + freq[2] + freq[4], 0.0);
}

TEST(Sampler, TopPKeepsTheSmallestSetReachingP) {
  // 0.5 + 0.25 reaches 0.7, so two tokens stay; 0.5 alone reaches 0.5.
  const std::vector<double> two =
      frequencies({.temperature = 1.0f, .top_p = 0.7f, .seed = 4}, 40000);
  EXPECT_NEAR(two[1], 0.5 / 0.75, 0.01);
  EXPECT_EQ(two[0] + two[2] + two[4], 0.0);
  const std::vector<double> one =
      frequencies({.temperature = 1.0f, .top_p = 0.5f, .seed = 5}, 1000);
  EXPECT_NEAR(one[1], 1.0, 1e-9);
}

TEST(Sampler, SameSeedSameStream) {
  Sampler a({.temperature = 1.0f, .seed = 42}), b({.temperature = 1.0f, .seed = 42});
  Sampler c({.temperature = 1.0f, .seed = 43});
  bool differs = false;
  for (int i = 0; i < 200; ++i) {
    const std::int32_t token = a.sample(kLogits);
    EXPECT_EQ(token, b.sample(kLogits));
    differs = differs || token != c.sample(kLogits);
  }
  EXPECT_TRUE(differs);
}

TEST(Sampler, RejectsInvalidSettings) {
  EXPECT_THROW(Sampler({.top_k = -1}), kvb::Error);
  EXPECT_THROW(Sampler({.top_p = 0.0f}), kvb::Error);
  EXPECT_THROW(Sampler({.top_p = 1.5f}), kvb::Error);
  Sampler sampler({.temperature = 1.0f});
  EXPECT_THROW(sampler.sample({}), kvb::Error);
}

}  // namespace
