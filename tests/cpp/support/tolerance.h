#pragma once

#include <cmath>
#include <cstddef>
#include <span>

// The C++ twins of python/kvbound/tolerance.py. Sums run in double so the metric adds no error.
namespace kvb::test {

// RMS of the error divided by RMS of the expected values.
inline double rel_rms(std::span<const float> actual, std::span<const float> expected) {
  double err = 0.0, ref = 0.0;
  for (std::size_t i = 0; i < expected.size(); ++i) {
    const double d = static_cast<double>(actual[i]) - expected[i];
    err += d * d;
    ref += static_cast<double>(expected[i]) * expected[i];
  }
  return std::sqrt(err / ref);
}

inline double cosine(std::span<const float> actual, std::span<const float> expected) {
  double dot = 0.0, aa = 0.0, ee = 0.0;
  for (std::size_t i = 0; i < expected.size(); ++i) {
    dot += static_cast<double>(actual[i]) * expected[i];
    aa += static_cast<double>(actual[i]) * actual[i];
    ee += static_cast<double>(expected[i]) * expected[i];
  }
  return dot / std::sqrt(aa * ee);
}

}  // namespace kvb::test
