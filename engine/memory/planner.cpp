#include "kvbound/memory_plan.h"

#include <algorithm>
#include <numeric>

namespace kvb {

std::vector<Lifetime> value_lifetimes(const Program& program) {
  const int n_ops = static_cast<int>(program.ops.size());
  std::vector<Lifetime> lifetimes(program.values.size(), Lifetime{n_ops, -1});
  for (int i = 0; i < n_ops; ++i) {
    const Op& op = program.ops[i];
    for (const int value : {op.in0, op.in1, op.out}) {
      if (value < 0) continue;
      lifetimes[value].first = std::min(lifetimes[value].first, i);
      lifetimes[value].last = std::max(lifetimes[value].last, i);
    }
  }
  // The host writes the input before the first op and reads the outputs after the last.
  if (program.input >= 0) lifetimes[program.input].first = 0;
  for (const int value : {program.logits, program.next_ids}) {
    if (value >= 0) lifetimes[value].last = n_ops - 1;
  }
  return lifetimes;
}

std::vector<std::size_t> value_bytes(const Program& program, std::int64_t max_tokens,
                                     std::int64_t max_outputs) {
  std::vector<std::size_t> bytes;
  bytes.reserve(program.values.size());
  for (const Value& value : program.values) {
    const std::int64_t rows = row_count(value.rows, max_tokens, max_outputs);
    const std::size_t raw = static_cast<std::size_t>(rows * value.cols) * size_bytes(value.dtype);
    bytes.push_back((raw + kArenaAlign - 1) / kArenaAlign * kArenaAlign);
  }
  return bytes;
}

namespace {

bool overlaps(const Lifetime& a, const Lifetime& b) {
  return a.first <= b.last && b.first <= a.last;
}

MemoryPlan plan_naive(std::span<const std::size_t> bytes) {
  MemoryPlan plan;
  for (const std::size_t size : bytes) {
    plan.offsets.push_back(plan.arena_bytes);
    plan.arena_bytes += size;
  }
  return plan;
}

// Places values from largest to smallest. Each goes into the lowest gap, among the values
// already placed that are alive at the same time, that is big enough to hold it.
MemoryPlan plan_greedy_by_size(std::span<const std::size_t> bytes,
                               std::span<const Lifetime> lifetimes) {
  std::vector<int> order(bytes.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return bytes[a] > bytes[b]; });

  MemoryPlan plan;
  plan.offsets.assign(bytes.size(), 0);
  std::vector<int> placed;  // kept sorted by offset
  for (const int value : order) {
    std::size_t offset = 0;
    for (const int other : placed) {
      if (!overlaps(lifetimes[value], lifetimes[other])) continue;
      if (plan.offsets[other] >= offset + bytes[value]) break;  // the gap before `other` fits
      offset = std::max(offset, plan.offsets[other] + bytes[other]);
    }
    plan.offsets[value] = offset;
    plan.arena_bytes = std::max(plan.arena_bytes, offset + bytes[value]);
    const auto at = std::upper_bound(placed.begin(), placed.end(), value, [&](int a, int b) {
      return plan.offsets[a] < plan.offsets[b];
    });
    placed.insert(at, value);
  }
  return plan;
}

}  // namespace

MemoryPlan plan_memory(std::span<const std::size_t> bytes, std::span<const Lifetime> lifetimes,
                       PlanStrategy strategy) {
  return strategy == PlanStrategy::Naive ? plan_naive(bytes)
                                         : plan_greedy_by_size(bytes, lifetimes);
}

std::size_t peak_live_bytes(std::span<const std::size_t> bytes,
                            std::span<const Lifetime> lifetimes) {
  int n_ops = 0;
  for (const Lifetime& life : lifetimes) n_ops = std::max(n_ops, life.last + 1);
  std::size_t peak = 0;
  for (int op = 0; op < n_ops; ++op) {
    std::size_t live = 0;
    for (std::size_t v = 0; v < bytes.size(); ++v) {
      if (lifetimes[v].first <= op && op <= lifetimes[v].last) live += bytes[v];
    }
    peak = std::max(peak, live);
  }
  return peak;
}

}  // namespace kvb
