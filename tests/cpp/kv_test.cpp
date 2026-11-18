// KV-cache bookkeeping that needs no GPU: the block allocator, block sharing between sequences
// and the layout strides.

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <set>
#include <utility>
#include <vector>

#include "kvbound/block_allocator.h"
#include "kvbound/error.h"
#include "kvbound/kv_layout.h"

namespace {

TEST(BlockAllocator, HandsOutEveryBlockOnce) {
  kvb::BlockAllocator allocator(4);
  std::set<int> blocks;
  for (int i = 0; i < 4; ++i) blocks.insert(allocator.allocate());
  EXPECT_EQ(blocks, (std::set<int>{0, 1, 2, 3}));
  EXPECT_EQ(allocator.n_free(), 0);
  EXPECT_THROW(allocator.allocate(), kvb::Error);
}

TEST(BlockAllocator, ReusesFreedBlocksAndRejectsDoubleFree) {
  kvb::BlockAllocator allocator(3);
  const int a = allocator.allocate();
  const int b = allocator.allocate();
  allocator.free(a);
  EXPECT_EQ(allocator.n_free(), 2);
  EXPECT_THROW(allocator.free(a), kvb::Error);
  EXPECT_THROW(allocator.free(7), kvb::Error);
  EXPECT_EQ(allocator.allocate(), a);  // most recently freed first
  allocator.free(b);
  allocator.free(a);
  EXPECT_EQ(allocator.n_free(), 3);
}

TEST(BlockAllocator, CountsReferences) {
  kvb::BlockAllocator allocator(2);
  const int a = allocator.allocate();
  EXPECT_EQ(allocator.ref_count(a), 1);
  allocator.retain(a);
  allocator.retain(a);
  EXPECT_EQ(allocator.ref_count(a), 3);
  EXPECT_EQ(allocator.n_free(), 1);

  // The block stays allocated until its last reference is released.
  allocator.free(a);
  allocator.free(a);
  EXPECT_EQ(allocator.ref_count(a), 1);
  EXPECT_EQ(allocator.n_free(), 1);
  allocator.free(a);
  EXPECT_EQ(allocator.ref_count(a), 0);
  EXPECT_EQ(allocator.n_free(), 2);

  EXPECT_THROW(allocator.free(a), kvb::Error);
  EXPECT_THROW(allocator.retain(a), kvb::Error);
  EXPECT_THROW(allocator.retain(2), kvb::Error);
  EXPECT_THROW(allocator.ref_count(2), kvb::Error);
  EXPECT_THROW(allocator.ref_count(-1), kvb::Error);
  // A reused block starts with one reference again.
  EXPECT_EQ(allocator.allocate(), a);
  EXPECT_EQ(allocator.ref_count(a), 1);
}

TEST(BlockSharing, ForkSharesTheBlocksThatHoldTokens) {
  kvb::BlockAllocator allocator(8);
  // Three blocks of 4 positions; the third is reserved and still empty.
  const std::vector<int> parent{allocator.allocate(), allocator.allocate(), allocator.allocate()};
  const std::vector<int> child = kvb::fork_blocks(allocator, parent, 6, 4);
  EXPECT_EQ(child, (std::vector<int>{parent[0], parent[1]}));
  EXPECT_EQ(allocator.ref_count(parent[0]), 2);
  EXPECT_EQ(allocator.ref_count(parent[1]), 2);
  EXPECT_EQ(allocator.ref_count(parent[2]), 1);
  EXPECT_EQ(allocator.n_free(), 5);  // a fork allocates nothing
  // 8 tokens fill two blocks exactly; no tokens share nothing.
  EXPECT_EQ(kvb::fork_blocks(allocator, parent, 8, 4).size(), 2u);
  EXPECT_TRUE(kvb::fork_blocks(allocator, parent, 0, 4).empty());
  EXPECT_EQ(allocator.ref_count(parent[1]), 3);

  // A fork that cannot be made changes no count.
  EXPECT_THROW(kvb::fork_blocks(allocator, parent, 13, 4), kvb::Error);
  EXPECT_THROW(kvb::fork_blocks(allocator, parent, 4, 0), kvb::Error);
  const std::vector<int> stale{parent[0], 7};
  EXPECT_THROW(kvb::fork_blocks(allocator, stale, 8, 4), kvb::Error);
  EXPECT_EQ(allocator.ref_count(parent[0]), 3);
}

TEST(BlockSharing, SharedBlockSurvivesOneOwnersFree) {
  kvb::BlockAllocator allocator(4);
  std::vector<int> parent{allocator.allocate(), allocator.allocate()};
  const std::vector<int> blocks = parent;
  std::vector<int> child = kvb::fork_blocks(allocator, parent, 8, 4);

  kvb::release_blocks(allocator, parent);
  EXPECT_TRUE(parent.empty());
  EXPECT_EQ(allocator.n_free(), 2);
  EXPECT_EQ(allocator.ref_count(blocks[0]), 1);
  EXPECT_EQ(allocator.ref_count(blocks[1]), 1);
  // Releasing the emptied list again is harmless: a list cannot free its blocks twice.
  kvb::release_blocks(allocator, parent);
  EXPECT_EQ(allocator.ref_count(blocks[0]), 1);

  kvb::release_blocks(allocator, child);
  EXPECT_EQ(allocator.n_free(), 4);
  EXPECT_THROW(allocator.free(blocks[0]), kvb::Error);
  // A list that names released blocks is refused.
  std::vector<int> stale = blocks;
  EXPECT_THROW(kvb::prepare_write(allocator, stale, 0, 4), kvb::Error);
  EXPECT_THROW(kvb::release_blocks(allocator, stale), kvb::Error);
  EXPECT_EQ(allocator.n_free(), 4);
}

TEST(BlockSharing, WritingCopiesASharedBlockOnce) {
  kvb::BlockAllocator allocator(8);
  std::vector<int> parent{allocator.allocate(), allocator.allocate()};
  const int first = parent[0], last = parent[1];
  // An unshared block is written in place.
  EXPECT_EQ(kvb::prepare_write(allocator, parent, 5, 4), std::nullopt);
  EXPECT_EQ(parent, (std::vector<int>{first, last}));

  // Three children of a parent with 6 tokens; the parent then goes away.
  std::vector<std::vector<int>> children;
  for (int i = 0; i < 3; ++i) children.push_back(kvb::fork_blocks(allocator, parent, 6, 4));
  kvb::release_blocks(allocator, parent);
  EXPECT_EQ(allocator.ref_count(last), 3);

  // Each child writes position 6. All but the last to do so copy the block; the last owns it.
  int copies = 0;
  for (std::vector<int>& child : children) {
    const int free_before = allocator.n_free();
    const std::optional<kvb::BlockCopy> copy = kvb::prepare_write(allocator, child, 6, 4);
    if (copy.has_value()) {
      ++copies;
      EXPECT_EQ(*copy, (kvb::BlockCopy{.src = last, .dst = child[1]}));
      EXPECT_NE(child[1], last);
      EXPECT_EQ(allocator.n_free(), free_before - 1);
    } else {
      EXPECT_EQ(child[1], last);
      EXPECT_EQ(allocator.n_free(), free_before);
    }
    EXPECT_EQ(allocator.ref_count(child[1]), 1);
    EXPECT_EQ(child[0], first);  // the block nobody writes stays shared
  }
  EXPECT_EQ(copies, 2);
  EXPECT_EQ(allocator.ref_count(first), 3);
  // A block is copied once: later writes to it are in place.
  for (std::vector<int>& child : children) {
    EXPECT_EQ(kvb::prepare_write(allocator, child, 7, 4), std::nullopt);
  }

  // Position 8 opens a new block, with nothing to copy.
  const int free_before = allocator.n_free();
  EXPECT_EQ(kvb::prepare_write(allocator, children[0], 8, 4), std::nullopt);
  EXPECT_EQ(children[0].size(), 3u);
  EXPECT_EQ(allocator.n_free(), free_before - 1);
  EXPECT_THROW(kvb::prepare_write(allocator, children[1], 12, 4), kvb::Error);
  EXPECT_THROW(kvb::prepare_write(allocator, children[1], -1, 4), kvb::Error);

  for (std::vector<int>& child : children) kvb::release_blocks(allocator, child);
  EXPECT_EQ(allocator.n_free(), 8);
}

TEST(BlockSharing, AFullPoolRefusesTheCopyAndKeepsTheList) {
  kvb::BlockAllocator allocator(1);
  std::vector<int> parent{allocator.allocate()};
  std::vector<int> child = kvb::fork_blocks(allocator, parent, 2, 4);
  EXPECT_THROW(kvb::prepare_write(allocator, child, 2, 4), kvb::Error);
  EXPECT_EQ(child, parent);
  EXPECT_EQ(allocator.ref_count(child[0]), 2);
  // Once the parent is gone the child owns the block and needs no copy.
  kvb::release_blocks(allocator, parent);
  EXPECT_EQ(kvb::prepare_write(allocator, child, 2, 4), std::nullopt);
}

TEST(BlockSharing, BeamsOfARequestCountTheCrossBlock) {
  // The cross-attention cache of a request is one block of encoder positions that no beam
  // writes: its count is the number of beams pointing at it.
  const int n_ctx = 64;
  kvb::BlockAllocator cross(2);
  std::vector<std::vector<int>> beams{{cross.allocate()}};
  const int block = beams[0][0];
  for (int i = 1; i < 5; ++i) beams.push_back(kvb::fork_blocks(cross, beams[0], n_ctx, n_ctx));
  EXPECT_EQ(cross.n_free(), 1);
  for (std::size_t i = 0; i < beams.size(); ++i) {
    EXPECT_EQ(cross.ref_count(block), static_cast<int>(beams.size() - i));
    kvb::release_blocks(cross, beams[i]);
  }
  EXPECT_EQ(cross.n_free(), 2);
}

// Beams fork, write and end at random, in the order kvbound/block_allocator.h gives for a beam
// step, against the pool a device would hold: here an array of token ids per block. The copies
// prepare_write returns are the only ones performed.
TEST(BlockSharing, ForkWriteFreeCycleKeepsEveryBeamsTokens) {
  const int n_beams = 5, block_size = 4, max_tokens = 30;
  const int blocks_per_seq = (max_tokens + block_size - 1) / block_size;
  // The smallest pool the header promises is enough.
  kvb::BlockAllocator allocator(n_beams * blocks_per_seq);
  std::vector<std::vector<std::int32_t>> pool(allocator.n_blocks(),
                                              std::vector<std::int32_t>(block_size, -1));
  std::uint32_t state = 12345;
  const auto random = [&](std::size_t n) {
    state = state * 1664525u + 1013904223u;
    return static_cast<std::size_t>(state >> 8) % n;
  };

  for (int request = 0; request < 20; ++request) {
    // One beam holds the 3-token prompt; every later step forks n_beams beams from random
    // parents.
    std::vector<std::vector<int>> lists(1);
    std::vector<std::vector<std::int32_t>> tokens(1);
    int copies = 0;
    for (int n = 0; n < max_tokens; ++n) {
      std::set<std::size_t> parents;
      if (n >= 3) {
        std::vector<std::vector<int>> next_lists;
        std::vector<std::vector<std::int32_t>> next_tokens;
        for (int i = 0; i < n_beams; ++i) {
          const std::size_t parent = random(lists.size());
          parents.insert(parent);
          next_lists.push_back(kvb::fork_blocks(allocator, lists[parent], n, block_size));
          next_tokens.push_back(tokens[parent]);
        }
        for (std::vector<int>& list : lists) kvb::release_blocks(allocator, list);
        lists = std::move(next_lists);
        tokens = std::move(next_tokens);
      }

      // Every beam writes position n. The children of one parent share its last block and all
      // but one of them copy it; a position that opens a new block copies nothing.
      int step_copies = 0;
      for (std::vector<int>& list : lists) {
        const std::optional<kvb::BlockCopy> copy =
            kvb::prepare_write(allocator, list, n, block_size);
        if (!copy.has_value()) continue;
        pool[copy->dst] = pool[copy->src];
        ++step_copies;
      }
      const bool shared = n >= 3 && n % block_size != 0;
      EXPECT_EQ(step_copies, shared ? n_beams - static_cast<int>(parents.size()) : 0)
          << "position " << n;
      copies += step_copies;

      for (std::size_t beam = 0; beam < lists.size(); ++beam) {
        const std::int32_t token = static_cast<std::int32_t>(random(1000));
        ASSERT_EQ(allocator.ref_count(lists[beam][n / block_size]), 1);
        pool[lists[beam][n / block_size]][n % block_size] = token;
        tokens[beam].push_back(token);
      }
      // Every beam reads back its own history through its block list.
      for (std::size_t beam = 0; beam < lists.size(); ++beam) {
        for (int p = 0; p <= n; ++p) {
          ASSERT_EQ(pool[lists[beam][p / block_size]][p % block_size], tokens[beam][p])
              << "beam " << beam << " position " << p << " after writing " << n;
        }
      }
    }
    EXPECT_GT(copies, 0);
    for (std::vector<int>& list : lists) kvb::release_blocks(allocator, list);
    // Every block returns, whatever was shared.
    ASSERT_EQ(allocator.n_free(), allocator.n_blocks());
    for (int block = 0; block < allocator.n_blocks(); ++block) {
      ASSERT_EQ(allocator.ref_count(block), 0);
    }
  }
}

// Every layout must place each element of K[layer][block][head][pos][dim] at its own offset
// inside the pool.
TEST(KvLayout, StridesCoverThePoolExactlyOnce) {
  const kvb::KvDims dims{
      .n_layers = 3, .n_blocks = 4, .n_kv_heads = 2, .block_size = 5, .head_dim = 6};
  for (const kvb::KvLayout layout :
       {kvb::KvLayout::HeadMajor, kvb::KvLayout::TokenMajor, kvb::KvLayout::DimMajor}) {
    SCOPED_TRACE(kvb::kv_layout_name(layout));
    const kvb::KvStrides s = kvb::kv_strides(layout, dims);
    std::vector<bool> seen(dims.elements(), false);
    for (std::int64_t layer = 0; layer < dims.n_layers; ++layer)
      for (std::int64_t block = 0; block < dims.n_blocks; ++block)
        for (std::int64_t head = 0; head < dims.n_kv_heads; ++head)
          for (std::int64_t pos = 0; pos < dims.block_size; ++pos)
            for (std::int64_t dim = 0; dim < dims.head_dim; ++dim) {
              const std::int64_t offset =
                  layer * s.layer + block * s.block + head * s.head + pos * s.pos + dim * s.dim;
              ASSERT_GE(offset, 0);
              ASSERT_LT(offset, dims.elements());
              ASSERT_FALSE(seen[offset]);
              seen[offset] = true;
            }
    EXPECT_EQ(kvb::kv_layout_from_name(kvb::kv_layout_name(layout)), layout);
  }
  EXPECT_FALSE(kvb::kv_layout_from_name("row_major").has_value());
}

TEST(KvLayout, BytesPerToken) {
  // Llama 3.2 1B in FP16: 16 layers, 8 KV heads, head dimension 64.
  EXPECT_EQ(kvb::kv_bytes_per_token(16, 8, 64, kvb::DType::F16), 32u * 1024u);
  // An INT8 vector is 64 bytes plus a 2-byte scale.
  EXPECT_EQ(kvb::kv_bytes_per_token(16, 8, 64, kvb::DType::I8), 2u * 16u * 8u * 66u);
}

}  // namespace
