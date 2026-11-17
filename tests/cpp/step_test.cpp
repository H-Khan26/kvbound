// Splitting a step's rows into the tiles prefill attention runs one block per.

#include "kvbound/step.h"

#include <gtest/gtest.h>

#include <vector>

#include "kvbound/error.h"

namespace {

using kvb::AttnTile;
using kvb::plan_attention_tiles;
using kvb::StepRow;

// Appends rows of `seq` at positions first, first + 1, ..., first + n - 1.
void add_run(std::vector<StepRow>& rows, int seq, int first, int n) {
  for (int i = 0; i < n; ++i) rows.push_back({.token_id = 0, .position = first + i, .seq = seq});
}

std::vector<std::pair<int, int>> as_pairs(const std::vector<AttnTile>& tiles) {
  std::vector<std::pair<int, int>> pairs;
  for (const AttnTile& tile : tiles) pairs.emplace_back(tile.first_row, tile.n_rows);
  return pairs;
}

TEST(PlanAttentionTiles, EmptyStepHasNoTiles) { EXPECT_TRUE(plan_attention_tiles({}, 64).empty()); }

TEST(PlanAttentionTiles, MixedStep) {
  std::vector<StepRow> rows;
  add_run(rows, 0, 0, 5);    // rows 0-4: a prompt
  add_run(rows, 1, 40, 1);   // row 5: decode
  add_run(rows, 2, 7, 1);    // row 6: decode
  add_run(rows, 3, 16, 10);  // rows 7-16: a chunk longer than max_rows, from position 16
  add_run(rows, 1, 41, 1);   // row 17: seq 1 again, consecutive with row 5 but not adjacent
  add_run(rows, 0, 5, 2);    // rows 18-19: seq 0 continues where rows 0-4 ended
  add_run(rows, 0, 9, 2);    // rows 20-21: seq 0 again, after a gap in positions
  const std::vector<std::pair<int, int>> expected = {
      {0, 4}, {4, 1}, {5, 1}, {6, 1}, {7, 4}, {11, 4}, {15, 2}, {17, 1}, {18, 2}, {20, 2},
  };
  EXPECT_EQ(as_pairs(plan_attention_tiles(rows, 4)), expected);
}

TEST(PlanAttentionTiles, SameSequenceOutOfOrderSplits) {
  std::vector<StepRow> rows;
  add_run(rows, 0, 3, 1);
  add_run(rows, 0, 2, 1);  // an earlier position of the same sequence
  add_run(rows, 0, 3, 1);
  add_run(rows, 0, 3, 1);  // a repeated position
  const std::vector<std::pair<int, int>> expected = {{0, 1}, {1, 2}, {3, 1}};
  EXPECT_EQ(as_pairs(plan_attention_tiles(rows, 64)), expected);
}

TEST(PlanAttentionTiles, LongPrefillFillsWholeTiles) {
  std::vector<StepRow> rows;
  add_run(rows, 2, 0, 130);
  const std::vector<std::pair<int, int>> expected = {{0, 64}, {64, 64}, {128, 2}};
  EXPECT_EQ(as_pairs(plan_attention_tiles(rows, 64)), expected);
  // One row per tile.
  EXPECT_EQ(plan_attention_tiles(std::span(rows).first(3), 1).size(), 3u);
}

TEST(PlanAttentionTiles, RejectsEmptyTiles) {
  const std::vector<StepRow> rows(1);
  EXPECT_THROW(plan_attention_tiles(rows, 0), kvb::Error);
}

}  // namespace
