#pragma once

#include <optional>
#include <span>
#include <vector>

namespace kvb {

// Hands out block ids 0..n-1 from a free list. Pure bookkeeping: it owns no memory, so it runs
// and is tested without a GPU. The KV pool uses one for cache blocks; the engine uses another
// for block-table rows.
//
// A block is reference-counted so that sequences with a common prefix can share it. allocate()
// returns a block with one reference, retain() adds one and free() releases one; the block
// returns to the free list when its last reference is released. A block that is never retained
// behaves as if there were no counts.
class BlockAllocator {
 public:
  explicit BlockAllocator(int n_blocks);

  int n_blocks() const noexcept { return static_cast<int>(refs_.size()); }
  // Blocks nobody references.
  int n_free() const noexcept { return static_cast<int>(free_.size()); }

  // Throws kvb::Error when no block is free.
  int allocate();
  // Adds a reference. Throws kvb::Error when the block is not currently allocated.
  void retain(int block);
  // Releases a reference. Throws kvb::Error when the block is not currently allocated.
  void free(int block);
  // References held on the block: 0 when it is free. Throws kvb::Error when there is no such
  // block.
  int ref_count(int block) const;

 private:
  void check(int block, const char* action) const;

  std::vector<int> free_;  // stack; the most recently freed block is reused first
  std::vector<int> refs_;
};

// --- Sharing blocks between sequences -------------------------------------------------------
//
// A sequence owns a block list: its physical cache blocks in logical order, block i holding
// positions [i * block_size, (i + 1) * block_size). Each entry of a list is one reference. The
// functions below decide every fork and copy on the CPU; the device only points block-table
// rows at the lists (KvPool::set_blocks) and performs the copies they return.
//
// Beam search uses them like this after BeamSearch::step() returns the next beams, where n is
// the number of tokens every old beam has cached:
//   1. next[i] = fork_blocks(allocator, old[step.beams[i].parent], n, block_size) for every i
//   2. release_blocks(allocator, old[b]) for every old beam b
//   3. for every i: prepare_write(allocator, next[i], n, block_size); then set_blocks(row i,
//      next[i])
//   4. perform every copy step 3 returned, then the step that writes position n, in that order
//      on one stream: a copy reads its source before any beam writes into it
// In that order a parent with one child hands its blocks over without a copy, a parent with c
// children costs c - 1 block copies (none when position n opens a new block), and a pool of
// num_beams * ceil(max_tokens / block_size) blocks never runs dry: no block is allocated
// before the old beams' references are dropped, and each one prepare_write allocates stays.
// Before the first step the beam holding the prompt calls prepare_write for every prompt
// position, which appends a block at each block boundary.
//
// Releasing a sequence is always release_blocks (or free() per block), whether or not its blocks
// are shared: it drops references, and a block returns to the pool with its last one. The
// Scheduler counts blocks per sequence and knows nothing of sharing or of blocks taken outside
// its plans. A caller that forks sequences from the allocator the Scheduler plans for reserves
// their worst case, num_beams * ceil(max_tokens / block_size), with it.
//
// The cross-attention cache of a speech request is one block that every beam of the request
// reads and nobody writes after the encoder. It is counted like any other block, in the cross
// cache's own allocator: the beam that ran the encoder holds the first reference, every fork
// retains the block for the child (retain(), or fork_blocks over the one-block list), every
// released beam frees it, and no beam calls prepare_write on it. So its count is the number of
// beams pointing at it, and it returns to the pool with the last beam of the request.

// One block the device copies whole, in every layer: keys, values and their scales.
struct BlockCopy {
  int src = 0;
  int dst = 0;

  bool operator==(const BlockCopy&) const = default;
};

// The block list of a child that starts with the first n_tokens tokens of its parent: the
// parent's blocks that hold them, each with one more reference. Blocks the parent holds beyond
// them are not shared. Throws kvb::Error when the parent has no block for one of the tokens.
std::vector<int> fork_blocks(BlockAllocator& allocator, std::span<const int> parent, int n_tokens,
                             int block_size);

// Releases one reference on every block of the list and empties it.
void release_blocks(BlockAllocator& allocator, std::vector<int>& blocks);

// Makes the block holding `position` writable for the owner of `blocks`; call it before every
// step in which the sequence writes that position.
//   * The list ends just before that block: a new block is appended.
//   * The block has other references: a new block takes its place in the list, one reference
//     on the old block is released, and the copy to perform is returned. The whole block is
//     copied, so positions before `position` keep their contents.
//   * Otherwise nothing changes.
// Throws kvb::Error when no block is free (the list is then unchanged) or when the list ends
// more than one block short of the position.
std::optional<BlockCopy> prepare_write(BlockAllocator& allocator, std::vector<int>& blocks,
                                       int position, int block_size);

}  // namespace kvb
