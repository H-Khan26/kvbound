#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <span>
#include <vector>

#include "kvbound/cuda/device.h"
#include "kvbound/kv_layout.h"

namespace kvb::cuda {

// The KV cache: one preallocated pool of keys, one of values, per-vector scales when the
// element type is quantized, and the block table that maps each sequence's logical blocks to
// physical ones. The pool knows nothing about which blocks are in use; that bookkeeping stays
// on the CPU.
class KvPool {
 public:
  KvPool(const KvDims& dims, KvLayout layout, int max_seqs, int blocks_per_seq,
         DType dtype = DType::F16);

  const KvDims& dims() const noexcept { return dims_; }
  KvLayout layout() const noexcept { return layout_; }
  int blocks_per_seq() const noexcept { return blocks_per_seq_; }
  DType dtype() const noexcept { return dtype_; }
  std::size_t bytes() const noexcept {
    return keys_.bytes() + values_.bytes() + key_scales_.bytes() + value_scales_.bytes();
  }

  // The cache as kernels of one layer see it.
  KvView layer_view(int layer) const noexcept;

  // Points sequence row `seq` at these physical blocks, in logical order.
  void set_blocks(int seq, std::span<const int> blocks, cudaStream_t stream);

 private:
  KvDims dims_;
  KvLayout layout_;
  KvStrides strides_;
  int blocks_per_seq_;
  DType dtype_;
  DeviceBuffer keys_;
  DeviceBuffer values_;
  DeviceBuffer key_scales_;
  DeviceBuffer value_scales_;
  DeviceBuffer block_table_;
  PinnedBuffer block_table_host_;
};

}  // namespace kvb::cuda
