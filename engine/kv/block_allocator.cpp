#include "kvbound/block_allocator.h"

#include "kvbound/error.h"

namespace kvb {

BlockAllocator::BlockAllocator(int n_blocks) : in_use_(n_blocks, false) {
  // Reversed, so the first allocations come out as 0, 1, 2, ...
  for (int block = n_blocks - 1; block >= 0; --block) free_.push_back(block);
}

int BlockAllocator::allocate() {
  if (free_.empty()) throw Error("no free block");
  const int block = free_.back();
  free_.pop_back();
  in_use_[block] = true;
  return block;
}

void BlockAllocator::free(int block) {
  if (block < 0 || block >= n_blocks() || !in_use_[block]) {
    throw Error("freeing a block that is not allocated");
  }
  in_use_[block] = false;
  free_.push_back(block);
}

}  // namespace kvb
