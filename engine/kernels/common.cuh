#pragma once

// Shared by the kernel sources: launch geometry and warp/block reductions.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "kvbound/cuda/device.h"
#include "kvbound/cuda/kernels.h"

namespace kvb::kernels {

inline constexpr int kBlockThreads = 256;
inline constexpr int kWarpSize = 32;
inline constexpr int kBlockWarps = kBlockThreads / kWarpSize;

// Blocks of kBlockThreads needed for n one-element threads.
inline unsigned blocks_for(std::size_t n) {
  return static_cast<unsigned>((n + kBlockThreads - 1) / kBlockThreads);
}

// Sum of v over the 32 lanes of a warp, returned to every lane. Each step adds the value held
// by the lane whose index differs in one bit, so five register exchanges cover the warp without
// touching memory.
__device__ inline float warp_sum(float v) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, offset);
  }
  return v;
}

// Sum of v over every thread of the block, returned to every thread. `scratch` is shared memory
// with one float per warp.
__device__ inline float block_sum(float v, float* scratch) {
  const int n_warps = blockDim.x / kWarpSize;
  v = warp_sum(v);
  if ((threadIdx.x & (kWarpSize - 1)) == 0) scratch[threadIdx.x / kWarpSize] = v;
  __syncthreads();
  float total = 0.0f;
  for (int warp = 0; warp < n_warps; ++warp) total += scratch[warp];
  __syncthreads();  // every thread has read scratch before a caller reuses it
  return total;
}

inline const __half* as_half(const f16* p) { return reinterpret_cast<const __half*>(p); }
inline __half* as_half(f16* p) { return reinterpret_cast<__half*>(p); }

}  // namespace kvb::kernels
