#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "kvbound/program.h"

namespace kvb {

// Activation memory planning: every intermediate value of a forward pass gets a fixed offset in
// one arena, sized for the largest step, before anything runs. Two values may share bytes when
// their lifetimes do not overlap.

// First and last op (inclusive) that touch a value.
struct Lifetime {
  int first = 0;
  int last = 0;
};

inline constexpr std::size_t kArenaAlign = 256;

std::vector<Lifetime> value_lifetimes(const Program& program);

// Bytes of each value at the largest step, rounded up to kArenaAlign. A Whisper model plans its
// encoder and decoder programs separately: the encoder's max_tokens is the clips of one step
// times the encoder positions of a clip, and it has no outputs.
std::vector<std::size_t> value_bytes(const Program& program, std::int64_t max_tokens,
                                     std::int64_t max_outputs);

enum class PlanStrategy : std::uint8_t {
  Naive,         // every value gets its own bytes
  GreedyBySize,  // largest first, each at the lowest offset free for its whole lifetime
};

struct MemoryPlan {
  std::vector<std::size_t> offsets;  // per value
  std::size_t arena_bytes = 0;
};

MemoryPlan plan_memory(std::span<const std::size_t> bytes, std::span<const Lifetime> lifetimes,
                       PlanStrategy strategy);

// Largest total size of values alive during any one op. No plan can use less.
std::size_t peak_live_bytes(std::span<const std::size_t> bytes,
                            std::span<const Lifetime> lifetimes);

}  // namespace kvb
