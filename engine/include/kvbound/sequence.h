#pragma once

#include <cstdint>
#include <functional>
#include <span>
#include <vector>

#include "kvbound/block_allocator.h"
#include "kvbound/step.h"

namespace kvb {

// A sequence's tokens and its place in a paged KV cache, as an engine tracks them on the CPU.
struct CachedSequence {
  int row = -1;             // block-table row, -1 until blocks are allocated
  std::vector<int> blocks;  // physical blocks, in logical order
  std::vector<std::int32_t> tokens;
  int n_cached = 0;  // tokens whose keys and values are in the cache
};

// Returns a sequence's blocks and its block-table row to their allocators. It keeps its tokens
// and holds no cache until blocks are allocated again.
void release_cache(CachedSequence& s, BlockAllocator& blocks, BlockAllocator& rows);

// The rows of the step that runs a plan's tokens.
struct StepRows {
  std::vector<StepRow> rows;
  // The last row of every sequence whose known tokens are all processed by the step: the rows
  // that produce a token.
  std::vector<std::int32_t> out_rows;
  std::vector<int> emitting;      // sequence ids, parallel to out_rows
  std::vector<int> cached_after;  // per run: the sequence's n_cached once the step has run
};

// Lays out `runs` as step rows, each sequence's next tokens at their positions and cache slots.
// `sequence` returns the state of a sequence id. Throws kvb::Error if a run names tokens or
// cache blocks its sequence does not have.
StepRows build_step_rows(std::span<const StepPlan::Run> runs, int block_size,
                         const std::function<const CachedSequence&(int)>& sequence);

}  // namespace kvb
