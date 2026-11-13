#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

namespace kvb {

// Order of the KV pool's dimensions in memory, outermost first. Every layout stores the same
// tensor K[layer][block][kv_head][pos][dim]; they differ in which elements end up adjacent, and
// so in how many memory sectors one warp touches per load.
enum class KvLayout : std::uint8_t {
  HeadMajor,   // layer, block, head, pos, dim: one head's keys form a contiguous stream
  TokenMajor,  // block, pos, layer, head, dim: one token's vectors for every layer are adjacent
  DimMajor,    // layer, block, head, dim, pos: one dimension of every position is adjacent
};

std::string_view kv_layout_name(KvLayout layout) noexcept;
std::optional<KvLayout> kv_layout_from_name(std::string_view name) noexcept;

// Shape of one pool (keys or values). A block holds block_size consecutive positions of one
// sequence; the static design uses one block per sequence with block_size = max context. A
// sequence that spans several blocks needs a power-of-two block_size.
struct KvDims {
  std::int64_t n_layers = 0;
  std::int64_t n_blocks = 0;
  std::int64_t n_kv_heads = 0;
  std::int64_t block_size = 0;
  std::int64_t head_dim = 0;

  std::int64_t elements() const noexcept {
    return n_layers * n_blocks * n_kv_heads * block_size * head_dim;
  }
};

// Element offset of K[layer][block][head][pos][dim] is the dot product of the indices with
// these strides.
struct KvStrides {
  std::int64_t layer = 0;
  std::int64_t block = 0;
  std::int64_t head = 0;
  std::int64_t pos = 0;
  std::int64_t dim = 0;
};

KvStrides kv_strides(KvLayout layout, const KvDims& dims) noexcept;

// Bytes of keys plus values one cached token occupies across all layers.
constexpr std::size_t kv_bytes_per_token(std::int64_t n_layers, std::int64_t n_kv_heads,
                                         std::int64_t head_dim,
                                         std::size_t bytes_per_element) noexcept {
  return static_cast<std::size_t>(2 * n_layers * n_kv_heads * head_dim) * bytes_per_element;
}

// What attention and KV-write kernels see of the cache, for one layer. Plain data, passed to
// kernels by value. A sequence's position p lives in physical block
// block_table[seq * blocks_per_seq + p / block_size] at offset p % block_size.
struct KvView {
  void* k = nullptr;  // pool bases, already advanced to the layer
  void* v = nullptr;
  const std::int32_t* block_table = nullptr;
  std::int32_t blocks_per_seq = 0;
  std::int32_t block_size = 0;
  std::int32_t block_shift = 0;  // log2(block_size) when blocks_per_seq > 1, else unused
  std::int64_t block_stride = 0;
  std::int64_t head_stride = 0;
  std::int64_t pos_stride = 0;
  std::int64_t dim_stride = 0;
};

}  // namespace kvb
