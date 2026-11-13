#include "kvbound/cuda/kv_pool.h"

#include <algorithm>
#include <bit>

namespace kvb::cuda {

namespace {

// Bytes of the per-vector scales of one pool; zero for an element type that needs none.
std::size_t scale_bytes(const KvDims& dims, DType dtype) {
  if (!kv_dtype_supported(dtype)) throw Error("the KV cache holds F16 or I8 elements");
  if (dtype != DType::I8) return 0;
  return static_cast<std::size_t>(dims.elements() / dims.head_dim) * size_bytes(DType::F16);
}

}  // namespace

KvPool::KvPool(const KvDims& dims, KvLayout layout, int max_seqs, int blocks_per_seq, DType dtype)
    : dims_(dims),
      layout_(layout),
      strides_(kv_strides(layout, dims)),
      blocks_per_seq_(blocks_per_seq),
      dtype_(dtype),
      keys_(static_cast<std::size_t>(dims.elements()) * size_bytes(dtype)),
      values_(keys_.bytes()),
      key_scales_(scale_bytes(dims, dtype)),
      value_scales_(key_scales_.bytes()),
      block_table_(static_cast<std::size_t>(max_seqs) * blocks_per_seq * sizeof(std::int32_t)),
      block_table_host_(block_table_.bytes()) {
  if (blocks_per_seq > 1 && !std::has_single_bit(static_cast<std::uint64_t>(dims.block_size))) {
    throw Error("a KV block size below the context length must be a power of two");
  }
  std::fill_n(block_table_host_.as<std::int32_t>(), max_seqs * blocks_per_seq, 0);
  KVB_CUDA(cudaMemset(block_table_.as(), 0, block_table_.bytes()));
}

KvView KvPool::layer_view(int layer) const noexcept {
  const std::int64_t base = layer * strides_.layer * static_cast<std::int64_t>(size_bytes(dtype_));
  const std::int64_t scales_per_block = dims_.n_kv_heads * dims_.block_size;
  const std::int64_t scale_base = layer * dims_.n_blocks * scales_per_block;
  const bool quantized = key_scales_.bytes() != 0;
  return KvView{
      .k = keys_.as<std::byte>() + base,
      .v = values_.as<std::byte>() + base,
      .block_table = block_table_.as<std::int32_t>(),
      .blocks_per_seq = blocks_per_seq_,
      .block_size = static_cast<std::int32_t>(dims_.block_size),
      .block_shift = std::countr_zero(static_cast<std::uint64_t>(dims_.block_size)),
      .block_stride = strides_.block,
      .head_stride = strides_.head,
      .pos_stride = strides_.pos,
      .dim_stride = strides_.dim,
      .dtype = dtype_,
      .k_scale = quantized ? key_scales_.as<std::uint16_t>() + scale_base : nullptr,
      .v_scale = quantized ? value_scales_.as<std::uint16_t>() + scale_base : nullptr,
      .scale_block_stride = scales_per_block,
      .scale_head_stride = dims_.block_size,
  };
}

void KvPool::set_blocks(int seq, std::span<const int> blocks, cudaStream_t stream) {
  if (static_cast<int>(blocks.size()) > blocks_per_seq_) {
    throw Error("sequence needs more blocks than the block table holds");
  }
  const std::size_t row = static_cast<std::size_t>(seq) * blocks_per_seq_;
  std::int32_t* host = block_table_host_.as<std::int32_t>() + row;
  std::copy(blocks.begin(), blocks.end(), host);
  // The copy reads the pinned row when the stream reaches it. Callers finish each step with a
  // stream synchronize, so no earlier copy is still pending when the row changes.
  KVB_CUDA(cudaMemcpyAsync(block_table_.as<std::int32_t>() + row, host,
                           blocks.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice, stream));
}

}  // namespace kvb::cuda
