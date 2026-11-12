#include "kvbound/cuda/kv_pool.h"

#include <algorithm>

namespace kvb::cuda {

KvPool::KvPool(const KvDims& dims, KvLayout layout, int max_seqs, int blocks_per_seq)
    : dims_(dims),
      layout_(layout),
      strides_(kv_strides(layout, dims)),
      blocks_per_seq_(blocks_per_seq),
      keys_(static_cast<std::size_t>(dims.elements()) * sizeof(std::uint16_t)),
      values_(keys_.bytes()),
      block_table_(static_cast<std::size_t>(max_seqs) * blocks_per_seq * sizeof(std::int32_t)),
      block_table_host_(block_table_.bytes()) {
  std::fill_n(block_table_host_.as<std::int32_t>(), max_seqs * blocks_per_seq, 0);
  KVB_CUDA(cudaMemset(block_table_.as(), 0, block_table_.bytes()));
}

KvView KvPool::layer_view(int layer) const noexcept {
  const std::int64_t base = layer * strides_.layer;
  return KvView{
      .k = keys_.as<std::uint16_t>() + base,
      .v = values_.as<std::uint16_t>() + base,
      .block_table = block_table_.as<std::int32_t>(),
      .blocks_per_seq = blocks_per_seq_,
      .block_size = static_cast<std::int32_t>(dims_.block_size),
      .block_stride = strides_.block,
      .head_stride = strides_.head,
      .pos_stride = strides_.pos,
      .dim_stride = strides_.dim,
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
