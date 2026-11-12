// Static batching, driven the way the engine drives it but with a stand-in that only counts.

#include "kvbound/scheduler.h"

#include <gtest/gtest.h>

#include <map>
#include <vector>

#include "kvbound/error.h"

namespace {

using kvb::Request;
using kvb::Scheduler;
using kvb::StepPlan;
using kvb::StepResult;

// Stands in for the engine: tracks how many tokens each sequence has and emits a token whenever
// a plan processes a sequence's last known token.
struct FakeEngine {
  struct Seq {
    int known = 0;
    int cached = 0;
  };
  std::map<int, Seq> seqs;
  std::map<int, int> prompts;
  std::int32_t next_token = 100;

  StepResult execute(const StepPlan& plan) {
    for (const int id : plan.free) EXPECT_EQ(seqs.erase(id), 1u);
    for (const StepPlan::Alloc& alloc : plan.alloc) seqs[alloc.seq] = {prompts.at(alloc.seq), 0};
    StepResult result;
    for (const StepPlan::Run& run : plan.run) {
      Seq& s = seqs.at(run.seq);
      s.cached += run.n_tokens;
      EXPECT_LE(s.cached, s.known);
      if (s.cached == s.known) {
        s.known += 1;
        result.tokens.push_back({run.seq, next_token++});
      }
    }
    return result;
  }
};

struct Trace {
  std::vector<StepPlan> plans;
  std::map<int, int> generated;
  std::vector<kvb::KvUsage> usage;
};

Trace drive(Scheduler& scheduler, const std::vector<Request>& requests) {
  FakeEngine engine;
  for (const Request& r : requests) {
    engine.prompts[r.id] = r.prompt_tokens;
    scheduler.submit(r);
  }
  Trace trace;
  while (!scheduler.done()) {
    const StepPlan plan = scheduler.next();
    const StepResult result = engine.execute(plan);
    if (!plan.run.empty()) scheduler.complete(result);
    for (const StepResult::Token& token : result.tokens) trace.generated[token.seq] += 1;
    trace.plans.push_back(plan);
    trace.usage.push_back(scheduler.usage());
    EXPECT_LT(trace.plans.size(), 10000u) << "scheduler does not terminate";
  }
  return trace;
}

int rows(const StepPlan& plan) {
  int n = 0;
  for (const StepPlan::Run& run : plan.run) n += run.n_tokens;
  return n;
}

TEST(StaticScheduler, EveryRequestGetsExactlyItsTokens) {
  Scheduler scheduler({.n_slots = 2, .max_ctx = 64, .block_size = 64, .max_step_tokens = 64});
  const std::vector<Request> requests = {{0, 5, 3}, {1, 9, 7}, {2, 1, 1}, {3, 20, 2}, {4, 4, 4}};
  const Trace trace = drive(scheduler, requests);
  for (const Request& r : requests) EXPECT_EQ(trace.generated.at(r.id), r.max_new_tokens);
  EXPECT_TRUE(scheduler.done());
  EXPECT_EQ(scheduler.usage().reserved, 0);
}

TEST(StaticScheduler, BatchRunsUntilItsLongestSequenceFinishes) {
  Scheduler scheduler({.n_slots = 2, .max_ctx = 64, .block_size = 64, .max_step_tokens = 64});
  const Trace trace = drive(scheduler, {{0, 4, 2}, {1, 4, 6}, {2, 4, 1}});

  // Plan 0 admits requests 0 and 1 and prefills both; request 2 waits.
  EXPECT_EQ(trace.plans[0].alloc.size(), 2u);
  EXPECT_EQ(rows(trace.plans[0]), 8);
  EXPECT_EQ(trace.usage[0].waiting, 1);
  // Request 0 finishes after its second token, yet request 2 is admitted only once request 1
  // has produced all six: prefill plus five decode steps later.
  EXPECT_EQ(trace.plans[1].run.size(), 2u);
  for (int step = 2; step <= 5; ++step) {
    EXPECT_EQ(trace.plans[step].run.size(), 1u) << step;
    EXPECT_TRUE(trace.plans[step].alloc.empty());
    EXPECT_EQ(trace.usage[step].waiting, 1);
  }
  EXPECT_EQ(trace.plans[6].free, (std::vector<int>{0, 1}));
  ASSERT_EQ(trace.plans[6].alloc.size(), 1u);
  EXPECT_EQ(trace.plans[6].alloc[0].seq, 2);
  // The last plan only returns request 2's slot.
  EXPECT_EQ(trace.plans.back().free, (std::vector<int>{2}));
  EXPECT_TRUE(trace.plans.back().run.empty());
}

TEST(StaticScheduler, ReservesAWholeSlotWhateverTheLength) {
  Scheduler scheduler({.n_slots = 4, .max_ctx = 100, .block_size = 100, .max_step_tokens = 100});
  const Trace trace = drive(scheduler, {{0, 3, 2}, {1, 50, 50}});
  // Two sequences hold 200 positions from admission on; after prefill only 53 are in use.
  EXPECT_EQ(trace.usage[0].reserved, 200);
  EXPECT_EQ(trace.usage[0].used, 53);
  EXPECT_EQ(trace.usage[0].running, 2);
  // Request 0 is done after its second token but its slot stays reserved.
  EXPECT_EQ(trace.usage[1].running, 1);
  EXPECT_EQ(trace.usage[1].reserved, 200);
}

TEST(StaticScheduler, SplitsPrefillAcrossStepsWithinTheRowBudget) {
  Scheduler scheduler({.n_slots = 2, .max_ctx = 64, .block_size = 16, .max_step_tokens = 8});
  const Trace trace = drive(scheduler, {{0, 20, 2}, {1, 3, 4}});
  for (const StepPlan& plan : trace.plans) EXPECT_LE(rows(plan), 8);
  // Request 0 needs 22 positions: two 16-token blocks. Request 1 needs one.
  EXPECT_EQ(trace.plans[0].alloc[0].n_blocks, 2);
  EXPECT_EQ(trace.plans[0].alloc[1].n_blocks, 1);
  // First step: 8 rows of request 0. Second: request 1 is not decoding yet, so request 0 takes
  // 8 more. Third: request 0's last 4, then request 1's 3.
  EXPECT_EQ(rows(trace.plans[0]), 8);
  EXPECT_EQ(trace.plans[0].run.size(), 1u);
  EXPECT_EQ(rows(trace.plans[2]), 7);
  EXPECT_EQ(trace.generated.at(0), 2);
  EXPECT_EQ(trace.generated.at(1), 4);
}

TEST(StaticScheduler, StopsASequenceAtEos) {
  Scheduler scheduler(
      {.n_slots = 1, .max_ctx = 64, .block_size = 64, .max_step_tokens = 64, .eos_token_id = 102});
  // The fake engine emits tokens 100, 101, 102, ...: the third is EOS.
  const Trace trace = drive(scheduler, {{0, 4, 10}});
  EXPECT_EQ(trace.generated.at(0), 3);
}

TEST(StaticScheduler, RejectsWhatCannotRun) {
  EXPECT_THROW(Scheduler({.n_slots = 4, .max_ctx = 8, .block_size = 8, .max_step_tokens = 2}),
               kvb::Error);
  EXPECT_THROW(Scheduler({.n_slots = 1, .max_ctx = 8, .block_size = 9, .max_step_tokens = 8}),
               kvb::Error);
  Scheduler scheduler({.n_slots = 1, .max_ctx = 8, .block_size = 8, .max_step_tokens = 8});
  EXPECT_THROW(scheduler.submit({0, 6, 3}), kvb::Error);
  EXPECT_THROW(scheduler.submit({0, 0, 3}), kvb::Error);
  scheduler.submit({0, 4, 4});
  const StepPlan plan = scheduler.next();
  EXPECT_THROW(scheduler.next(), kvb::Error);
  EXPECT_FALSE(plan.run.empty());
}

}  // namespace
