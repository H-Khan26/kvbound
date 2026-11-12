// KV-cache bookkeeping that needs no GPU: the block allocator and the layout strides.

#include <gtest/gtest.h>

#include <set>
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
  EXPECT_EQ(kvb::kv_bytes_per_token(16, 8, 64, 2), 32u * 1024u);
}

}  // namespace
