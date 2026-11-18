#include "kvbound/block_allocator.h"

#include <string>

#include "kvbound/error.h"

namespace kvb {

BlockAllocator::BlockAllocator(int n_blocks) : refs_(n_blocks, 0) {
  // Reversed, so the first allocations come out as 0, 1, 2, ...
  for (int block = n_blocks - 1; block >= 0; --block) free_.push_back(block);
}

void BlockAllocator::check(int block, const char* action) const {
  if (block < 0 || block >= n_blocks() || refs_[block] == 0) {
    throw Error(std::string(action) + " a block that is not allocated");
  }
}

int BlockAllocator::allocate() {
  if (free_.empty()) throw Error("no free block");
  const int block = free_.back();
  free_.pop_back();
  refs_[block] = 1;
  return block;
}

void BlockAllocator::retain(int block) {
  check(block, "retaining");
  ++refs_[block];
}

void BlockAllocator::free(int block) {
  check(block, "freeing");
  if (--refs_[block] == 0) free_.push_back(block);
}

int BlockAllocator::ref_count(int block) const {
  if (block < 0 || block >= n_blocks()) throw Error("no block " + std::to_string(block));
  return refs_[block];
}

std::vector<int> fork_blocks(BlockAllocator& allocator, std::span<const int> parent, int n_tokens,
                             int block_size) {
  if (n_tokens < 0 || block_size <= 0) throw Error("a fork needs n_tokens >= 0 and block_size > 0");
  const std::size_t n_shared = (static_cast<std::size_t>(n_tokens) + block_size - 1) / block_size;
  if (n_shared > parent.size()) throw Error("forking tokens the parent has no block for");
  // Checked before any count changes, so a throw leaves the allocator as it was.
  for (const int block : parent.first(n_shared)) {
    if (allocator.ref_count(block) == 0) throw Error("forking a block that is not allocated");
  }
  for (const int block : parent.first(n_shared)) allocator.retain(block);
  return {parent.begin(), parent.begin() + static_cast<std::ptrdiff_t>(n_shared)};
}

void release_blocks(BlockAllocator& allocator, std::vector<int>& blocks) {
  for (const int block : blocks) allocator.free(block);
  blocks.clear();
}

std::optional<BlockCopy> prepare_write(BlockAllocator& allocator, std::vector<int>& blocks,
                                       int position, int block_size) {
  if (position < 0 || block_size <= 0) throw Error("a write needs position >= 0, block_size > 0");
  const std::size_t index = static_cast<std::size_t>(position / block_size);
  if (index > blocks.size()) throw Error("writing past the block after the sequence's last");
  if (index == blocks.size()) {
    blocks.push_back(allocator.allocate());
    return std::nullopt;
  }
  const int shared = blocks[index];
  const int refs = allocator.ref_count(shared);
  if (refs == 0) throw Error("writing a block that is not allocated");
  if (refs == 1) return std::nullopt;
  const int own = allocator.allocate();
  allocator.free(shared);
  blocks[index] = own;
  return BlockCopy{.src = shared, .dst = own};
}

}  // namespace kvb
