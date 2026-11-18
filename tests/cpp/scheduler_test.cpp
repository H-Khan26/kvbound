// Both scheduling policies, driven the way the engine drives them but with a stand-in that only
// counts tokens and blocks.

#include "kvbound/scheduler.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <map>
#include <vector>

#include "kvbound/error.h"

namespace {

using kvb::Request;
using kvb::SchedulePolicy;
using kvb::Scheduler;
using kvb::StepPlan;
using kvb::StepResult;

// Stands in for the engine: tracks how many tokens and blocks each sequence has, checks that
// plans stay within the pool, and emits a token whenever a plan processes a sequence's last
// known token. With an encoder, it also checks that a sequence's audio is encoded once, while it
// holds blocks, before any of its tokens run.
struct FakeEngine {
  struct Seq {
    int known = 0;
    int cached = 0;
    int blocks = 0;
    bool encoded = false;
  };
  std::map<int, Seq> seqs;
  std::map<int, int> prompts;
  int block_size = 0;
  int free_blocks = 0;
  int max_encodes = 0;  // sequences one step may encode; 0 for a model without an encoder
  std::int32_t next_token = 100;

  StepResult execute(const StepPlan& plan) {
    for (const int id : plan.free) {
      free_blocks += seqs.at(id).blocks;
      EXPECT_EQ(seqs.erase(id), 1u);
    }
    for (const int id : plan.preempt) {
      Seq& s = seqs.at(id);
      free_blocks += s.blocks;
      s.blocks = 0;
      s.cached = 0;
      s.encoded = false;  // the cross-attention cache goes with the blocks
    }
    for (const StepPlan::Alloc& alloc : plan.alloc) {
      // A sequence is registered by its first allocation.
      Seq& s = seqs.try_emplace(alloc.seq, Seq{prompts.at(alloc.seq), 0, 0}).first->second;
      s.blocks += alloc.n_blocks;
      free_blocks -= alloc.n_blocks;
      EXPECT_GE(free_blocks, 0) << "plan allocates more blocks than the pool has";
    }
    EXPECT_LE(static_cast<int>(plan.encode.size()), max_encodes);
    for (const int id : plan.encode) {
      Seq& s = seqs.at(id);
      EXPECT_GT(s.blocks, 0) << "sequence " << id << " is encoded without blocks";
      EXPECT_FALSE(s.encoded) << "sequence " << id << " is encoded twice";
      s.encoded = true;
    }
    StepResult result;
    for (const StepPlan::Run& run : plan.run) {
      Seq& s = seqs.at(run.seq);
      if (max_encodes > 0) EXPECT_TRUE(s.encoded) << "sequence " << run.seq << " runs unencoded";
      s.cached += run.n_tokens;
      EXPECT_LE(s.cached, s.known);
      EXPECT_LE(s.cached, s.blocks * block_size) << "sequence runs past its blocks";
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

// Runs the requests to completion. `late` holds requests submitted just before the plan with
// that index instead of at the start.
Trace drive(Scheduler& scheduler, const std::vector<Request>& requests,
            const std::map<std::size_t, Request>& late = {}) {
  FakeEngine engine;
  engine.block_size = scheduler.config().block_size;
  engine.free_blocks = scheduler.config().n_blocks;
  if (scheduler.config().encode_rows > 0) {
    engine.max_encodes = scheduler.config().max_encodes_per_step;
  }
  for (const Request& r : requests) {
    engine.prompts[r.id] = r.prompt_tokens;
    scheduler.submit(r);
  }
  Trace trace;
  while (!scheduler.done() || late.lower_bound(trace.plans.size()) != late.end()) {
    if (const auto it = late.find(trace.plans.size()); it != late.end()) {
      engine.prompts[it->second.id] = it->second.prompt_tokens;
      scheduler.submit(it->second);
    }
    const StepPlan plan = scheduler.next();
    const StepResult result = engine.execute(plan);
    if (!plan.run.empty()) scheduler.complete(result);
    for (const StepResult::Token& token : result.tokens) trace.generated[token.seq] += 1;
    trace.plans.push_back(plan);
    trace.usage.push_back(scheduler.usage());
    EXPECT_LT(trace.plans.size(), 10000u) << "scheduler does not terminate";
    if (trace.plans.size() >= 10000u) break;
  }
  EXPECT_EQ(engine.free_blocks, scheduler.config().n_blocks);
  EXPECT_EQ(scheduler.free_blocks(), scheduler.config().n_blocks);
  return trace;
}

std::vector<int> allocations(const StepPlan& plan, int seq) {
  std::vector<int> blocks;
  for (const StepPlan::Alloc& alloc : plan.alloc) {
    if (alloc.seq == seq) blocks.push_back(alloc.n_blocks);
  }
  return blocks;
}

int rows(const StepPlan& plan) {
  int n = 0;
  for (const StepPlan::Run& run : plan.run) n += run.n_tokens;
  return n;
}

// How many plans list the sequence in `list`, which is StepPlan::encode or StepPlan::preempt.
int times(const Trace& trace, std::vector<int> StepPlan::* list, int seq) {
  int n = 0;
  for (const StepPlan& plan : trace.plans) {
    n += static_cast<int>(std::count((plan.*list).begin(), (plan.*list).end(), seq));
  }
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
  EXPECT_THROW(scheduler.submit({0, 3, 0}), kvb::Error);
  scheduler.submit({0, 4, 4});
  const StepPlan plan = scheduler.next();
  EXPECT_THROW(scheduler.next(), kvb::Error);
  EXPECT_FALSE(plan.run.empty());
}

Scheduler continuous(int n_slots, int max_ctx, int block_size, int n_blocks, int max_step_tokens) {
  return Scheduler({.policy = SchedulePolicy::Continuous,
                    .n_slots = n_slots,
                    .max_ctx = max_ctx,
                    .block_size = block_size,
                    .n_blocks = n_blocks,
                    .max_step_tokens = max_step_tokens});
}

TEST(ContinuousScheduler, AdmitsARequestAsSoonAsASequenceFinishes) {
  // The same requests as the static test above, with one block per sequence.
  Scheduler scheduler = continuous(2, 64, 64, 0, 64);
  const Trace trace = drive(scheduler, {{0, 4, 2}, {1, 4, 6}, {2, 4, 1}});

  EXPECT_EQ(trace.plans[0].alloc.size(), 2u);
  // Request 0 finishes in the second step. The third plan frees it and admits request 2, which
  // prefills next to request 1's decode row instead of waiting for request 1 to finish.
  EXPECT_EQ(trace.plans[2].free, (std::vector<int>{0}));
  EXPECT_EQ(allocations(trace.plans[2], 2), (std::vector<int>{1}));
  EXPECT_EQ(rows(trace.plans[2]), 5);
  EXPECT_EQ(trace.usage[2].waiting, 0);
  EXPECT_EQ(trace.generated.at(1), 6);
  EXPECT_EQ(trace.generated.at(2), 1);
  EXPECT_EQ(trace.plans.size(), 7u);
}

TEST(ContinuousScheduler, AllocatesBlocksAsASequenceGrows) {
  Scheduler scheduler = continuous(1, 32, 4, 0, 32);
  const Trace trace = drive(scheduler, {{0, 5, 6}});
  // Five prompt tokens take two 4-token blocks. The third block arrives with the ninth token.
  EXPECT_EQ(allocations(trace.plans[0], 0), (std::vector<int>{2}));
  for (int step = 1; step <= 3; ++step) EXPECT_TRUE(trace.plans[step].alloc.empty()) << step;
  EXPECT_EQ(allocations(trace.plans[4], 0), (std::vector<int>{1}));
  EXPECT_EQ(trace.usage[3].reserved, 8);
  EXPECT_EQ(trace.usage[3].used, 8);
  EXPECT_EQ(trace.usage[4].reserved, 12);
  EXPECT_EQ(trace.usage[4].used, 9);
  EXPECT_EQ(trace.generated.at(0), 6);
}

TEST(ContinuousScheduler, KeepsASpareBlockForEverySequenceThatCanGrow) {
  // Five 4-token blocks; each request needs one block for its prompt and three in the end.
  Scheduler scheduler = continuous(4, 16, 4, 5, 16);
  const Trace trace = drive(scheduler, {{0, 4, 8}, {1, 4, 8}, {2, 4, 8}});
  // Two requests leave three blocks free, enough for request 2's prompt. It still waits: with
  // three sequences growing the pool would need a fourth spare block.
  EXPECT_EQ(trace.plans[0].alloc.size(), 2u);
  EXPECT_EQ(trace.usage[0].waiting, 1);
  EXPECT_EQ(trace.usage[0].reserved, 8);
  for (const auto& [id, n] : trace.generated) EXPECT_EQ(n, 8) << id;
}

TEST(ContinuousScheduler, PreemptsTheYoungestSequenceWhenThePoolRunsDry) {
  Scheduler scheduler = continuous(2, 16, 4, 5, 16);
  const Trace trace = drive(scheduler, {{0, 4, 8}, {1, 4, 8}});
  // Both prompts fit, and each sequence takes a second block for its fifth token.
  EXPECT_EQ(trace.plans[1].alloc.size(), 2u);
  for (int step = 0; step <= 4; ++step) EXPECT_TRUE(trace.plans[step].preempt.empty()) << step;
  // At nine tokens each wants a third block and one is free. Request 0 is older and gets it;
  // request 1 gives up its two blocks and waits.
  EXPECT_EQ(allocations(trace.plans[5], 0), (std::vector<int>{1}));
  EXPECT_EQ(trace.plans[5].preempt, (std::vector<int>{1}));
  EXPECT_EQ(trace.plans[5].run.size(), 1u);
  EXPECT_EQ(trace.usage[5].running, 1);
  EXPECT_EQ(trace.usage[5].waiting, 1);
  EXPECT_EQ(trace.usage[5].reserved, 12);
  // Two free blocks do not hold request 1's nine tokens, so it stays out until request 0 ends.
  EXPECT_TRUE(trace.plans[6].alloc.empty());
  EXPECT_TRUE(trace.plans[7].alloc.empty());
  EXPECT_EQ(trace.plans[8].free, (std::vector<int>{0}));
  EXPECT_EQ(allocations(trace.plans[8], 1), (std::vector<int>{3}));
  // Readmitted, it recomputes all nine tokens it had in one prefill step, then decodes on.
  EXPECT_EQ(rows(trace.plans[8]), 9);
  EXPECT_EQ(trace.generated.at(0), 8);
  EXPECT_EQ(trace.generated.at(1), 8);
  int total_rows = 0;
  for (const StepPlan& plan : trace.plans) total_rows += rows(plan);
  EXPECT_EQ(total_rows, 2 * (4 + 7) + 8);  // eight of request 1's rows ran twice
}

TEST(ContinuousScheduler, EveryRequestGetsExactlyItsTokensInATightPool) {
  // Twelve 8-token blocks for requests that need up to eight each: preemption is unavoidable.
  Scheduler scheduler = continuous(4, 64, 8, 12, 16);
  const std::vector<Request> requests = {{0, 5, 3},  {1, 9, 30},  {2, 1, 1}, {3, 20, 12},
                                         {4, 4, 40}, {5, 30, 30}, {6, 3, 50}};
  const Trace trace = drive(scheduler, requests, {{3, {7, 40, 20}}, {200, {8, 2, 2}}});
  for (const Request& r : requests) EXPECT_EQ(trace.generated.at(r.id), r.max_new_tokens);
  EXPECT_EQ(trace.generated.at(7), 20);
  EXPECT_EQ(trace.generated.at(8), 2);
  int preemptions = 0;
  for (std::size_t i = 0; i < trace.plans.size(); ++i) {
    preemptions += static_cast<int>(trace.plans[i].preempt.size());
    EXPECT_LE(rows(trace.plans[i]), 16);
    EXPECT_LE(trace.usage[i].reserved, 12 * 8);
    EXPECT_LE(trace.usage[i].running, 4);
  }
  EXPECT_GT(preemptions, 0);
  EXPECT_TRUE(scheduler.done());
  EXPECT_EQ(scheduler.usage().reserved, 0);
}

TEST(ContinuousScheduler, AdmitsRequestsThatArriveWhileOthersRun) {
  Scheduler scheduler = continuous(2, 64, 16, 0, 64);
  // Request 1 is submitted before the fourth plan, while request 0 is decoding.
  const Trace trace = drive(scheduler, {{0, 4, 10}}, {{3, {1, 6, 2}}});
  EXPECT_EQ(allocations(trace.plans[3], 1), (std::vector<int>{1}));
  EXPECT_EQ(rows(trace.plans[3]), 1 + 6);
  EXPECT_EQ(trace.generated.at(1), 2);
}

TEST(Scheduler, RejectsAPoolTooSmallForItsPolicy) {
  // Static batching needs every slot's full context; continuous batching needs one.
  EXPECT_THROW(
      Scheduler(
          {.n_slots = 4, .max_ctx = 64, .block_size = 16, .n_blocks = 15, .max_step_tokens = 64}),
      kvb::Error);
  EXPECT_THROW(continuous(4, 64, 16, 3, 64), kvb::Error);
  EXPECT_NO_THROW(continuous(4, 64, 16, 4, 64));
}

// The speech path: every request carries audio that the encoder processes before its tokens run.
Scheduler encoding(SchedulePolicy policy, int n_slots, int max_ctx, int block_size, int n_blocks,
                   int max_step_tokens, int max_encodes_per_step) {
  return Scheduler({.policy = policy,
                    .n_slots = n_slots,
                    .max_ctx = max_ctx,
                    .block_size = block_size,
                    .n_blocks = n_blocks,
                    .max_step_tokens = max_step_tokens,
                    .encode_rows = 64,
                    .max_encodes_per_step = max_encodes_per_step});
}

TEST(EncodingScheduler, StaticBatchFillsAtOnceAndIsEncodedOneClipAPlan) {
  Scheduler scheduler = encoding(SchedulePolicy::Static, 3, 32, 32, 0, 32, 1);
  const Trace trace = drive(scheduler, {{0, 4, 5}, {1, 4, 5}, {2, 4, 5}, {3, 4, 1}});
  // The batch is admitted whole, as without an encoder, but its members start a plan apart:
  // each plan encodes one clip, and a sequence prefills in the plan that encodes it.
  EXPECT_EQ(trace.plans[0].alloc.size(), 3u);
  for (int step = 0; step < 3; ++step) {
    EXPECT_EQ(trace.plans[step].encode, (std::vector<int>{step})) << step;
    ASSERT_EQ(trace.plans[step].run.size(), static_cast<std::size_t>(step + 1)) << step;
    EXPECT_EQ(trace.plans[step].run.back().seq, step);
    EXPECT_EQ(trace.plans[step].run.back().n_tokens, 4);
    EXPECT_EQ(trace.usage[step].running, 3);
  }
  // Request 2 starts two plans late and ends two plans after request 0. Request 3 joins the
  // next batch and is encoded then.
  EXPECT_EQ(trace.plans[7].free, (std::vector<int>{0, 1, 2}));
  EXPECT_EQ(trace.plans[7].encode, (std::vector<int>{3}));
  for (int id = 0; id < 4; ++id) {
    EXPECT_EQ(times(trace, &StepPlan::encode, id), 1) << id;
    EXPECT_EQ(trace.generated.at(id), id == 3 ? 1 : 5) << id;
  }
}

TEST(EncodingScheduler, EncodesAsManyClipsAsAStepAllows) {
  Scheduler scheduler = encoding(SchedulePolicy::Continuous, 4, 32, 32, 0, 32, 2);
  const Trace trace = drive(scheduler, {{0, 4, 3}, {1, 4, 3}, {2, 4, 3}}, {{1, {3, 4, 3}}});
  EXPECT_EQ(trace.plans[0].encode, (std::vector<int>{0, 1}));
  EXPECT_EQ(rows(trace.plans[0]), 8);
  // Request 2 holds its block from the first plan and waits a plan for the encoder. Request 3
  // arrives before the second plan, which admits and encodes it.
  EXPECT_EQ(trace.plans[0].alloc.size(), 3u);
  EXPECT_EQ(allocations(trace.plans[1], 3), (std::vector<int>{1}));
  EXPECT_EQ(trace.plans[1].encode, (std::vector<int>{2, 3}));
  EXPECT_EQ(rows(trace.plans[1]), 1 + 1 + 4 + 4);
  for (std::size_t step = 2; step < trace.plans.size(); ++step) {
    EXPECT_TRUE(trace.plans[step].encode.empty()) << step;
  }
  for (int id = 0; id < 4; ++id) EXPECT_EQ(trace.generated.at(id), 3) << id;
}

TEST(EncodingScheduler, APreemptedSequenceIsEncodedAgain) {
  // The pool and requests of PreemptsTheYoungestSequenceWhenThePoolRunsDry.
  Scheduler scheduler = encoding(SchedulePolicy::Continuous, 2, 16, 4, 5, 16, 2);
  const Trace trace = drive(scheduler, {{0, 4, 8}, {1, 4, 8}});
  EXPECT_EQ(trace.plans[0].encode, (std::vector<int>{0, 1}));
  EXPECT_EQ(trace.plans[5].preempt, (std::vector<int>{1}));
  EXPECT_TRUE(trace.plans[5].encode.empty());
  // Readmitted, request 1 goes through the encoder again before it recomputes its nine tokens.
  EXPECT_EQ(allocations(trace.plans[8], 1), (std::vector<int>{3}));
  EXPECT_EQ(trace.plans[8].encode, (std::vector<int>{1}));
  EXPECT_EQ(rows(trace.plans[8]), 9);
  EXPECT_EQ(times(trace, &StepPlan::encode, 0), 1);
  EXPECT_EQ(times(trace, &StepPlan::encode, 1), 2);
  EXPECT_EQ(trace.generated.at(0), 8);
  EXPECT_EQ(trace.generated.at(1), 8);
}

TEST(EncodingScheduler, EncodesOncePerAdmissionInATightPool) {
  const std::vector<Request> requests = {{0, 5, 3},  {1, 9, 30},  {2, 1, 1}, {3, 20, 12},
                                         {4, 4, 40}, {5, 30, 30}, {6, 3, 50}};
  const std::map<std::size_t, Request> late = {{3, {7, 40, 20}}, {200, {8, 2, 2}}};
  for (const int max_encodes : {1, 4}) {
    SCOPED_TRACE(max_encodes);
    Scheduler scheduler = encoding(SchedulePolicy::Continuous, 4, 64, 8, 12, 16, max_encodes);
    const Trace trace = drive(scheduler, requests, late);
    int preemptions = 0;
    for (int id = 0; id <= 8; ++id) {
      const int preempted = times(trace, &StepPlan::preempt, id);
      EXPECT_EQ(times(trace, &StepPlan::encode, id), 1 + preempted) << id;
      preemptions += preempted;
    }
    EXPECT_GT(preemptions, 0);
    for (const Request& r : requests) EXPECT_EQ(trace.generated.at(r.id), r.max_new_tokens);
    EXPECT_EQ(trace.generated.at(7), 20);
    EXPECT_EQ(trace.generated.at(8), 2);
    EXPECT_TRUE(scheduler.done());

    // An encoder that takes every admitted clip at once changes nothing else about the plans.
    if (max_encodes < 4) continue;
    Scheduler plain = continuous(4, 64, 8, 12, 16);
    const Trace without = drive(plain, requests, late);
    ASSERT_EQ(trace.plans.size(), without.plans.size());
    for (std::size_t i = 0; i < trace.plans.size(); ++i) {
      EXPECT_TRUE(without.plans[i].encode.empty());
      EXPECT_EQ(trace.plans[i].free, without.plans[i].free) << i;
      EXPECT_EQ(trace.plans[i].preempt, without.plans[i].preempt) << i;
      EXPECT_EQ(rows(trace.plans[i]), rows(without.plans[i])) << i;
      EXPECT_EQ(trace.plans[i].alloc.size(), without.plans[i].alloc.size()) << i;
    }
  }
}

TEST(EncodingScheduler, RejectsAnEncoderThatTakesNoClips) {
  EXPECT_THROW(encoding(SchedulePolicy::Static, 1, 8, 8, 0, 8, 0), kvb::Error);
  EXPECT_THROW(
      Scheduler(
          {.n_slots = 1, .max_ctx = 8, .block_size = 8, .max_step_tokens = 8, .encode_rows = -1}),
      kvb::Error);
  // Without an encoder the limit is not read.
  EXPECT_NO_THROW(Scheduler({.n_slots = 1,
                             .max_ctx = 8,
                             .block_size = 8,
                             .max_step_tokens = 8,
                             .max_encodes_per_step = 0}));
}

}  // namespace
