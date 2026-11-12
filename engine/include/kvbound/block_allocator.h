#pragma once

#include <vector>

namespace kvb {

// Hands out block ids 0..n-1 from a free list. Pure bookkeeping: it owns no memory, so it runs
// and is tested without a GPU. The KV pool uses one for cache blocks; the engine uses another
// for block-table rows.
class BlockAllocator {
 public:
  explicit BlockAllocator(int n_blocks);

  int n_blocks() const noexcept { return static_cast<int>(in_use_.size()); }
  int n_free() const noexcept { return static_cast<int>(free_.size()); }

  // Throws kvb::Error when no block is free.
  int allocate();
  // Throws kvb::Error when the block is not currently allocated.
  void free(int block);

 private:
  std::vector<int> free_;  // stack; the most recently freed block is reused first
  std::vector<bool> in_use_;
};

}  // namespace kvb
