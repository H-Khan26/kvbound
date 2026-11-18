#include "kvbound/sequence.h"

#include <string>

#include "kvbound/error.h"

namespace kvb {

void release_cache(CachedSequence& s, BlockAllocator& blocks, BlockAllocator& rows) {
  for (const int block : s.blocks) blocks.free(block);
  s.blocks.clear();
  if (s.row >= 0) rows.free(s.row);
  s.row = -1;
  s.n_cached = 0;
}

StepRows build_step_rows(std::span<const StepPlan::Run> runs, int block_size,
                         const std::function<const CachedSequence&(int)>& sequence) {
  StepRows step;
  for (const StepPlan::Run& run : runs) {
    const CachedSequence& s = sequence(run.seq);
    const int end = s.n_cached + run.n_tokens;
    if (run.n_tokens <= 0 || end > static_cast<int>(s.tokens.size())) {
      throw Error("plan runs tokens sequence " + std::to_string(run.seq) + " does not have");
    }
    if (end > static_cast<int>(s.blocks.size()) * block_size) {
      throw Error("sequence " + std::to_string(run.seq) + " has no cache block for its tokens");
    }
    for (int pos = s.n_cached; pos < end; ++pos) {
      step.rows.push_back({.token_id = s.tokens[pos],
                           .position = pos,
                           .seq = s.row,
                           .kv_slot = s.blocks[pos / block_size] * block_size + pos % block_size});
    }
    step.cached_after.push_back(end);
    if (end == static_cast<int>(s.tokens.size())) {
      step.out_rows.push_back(static_cast<std::int32_t>(step.rows.size()) - 1);
      step.emitting.push_back(run.seq);
    }
  }
  return step;
}

}  // namespace kvb
