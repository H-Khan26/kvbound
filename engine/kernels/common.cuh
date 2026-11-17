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

// The FP16 value in the low 16 bits of a word, for kernels that load several at once.
__device__ inline float f16_bits_to_float(unsigned bits) {
  return __half2float(__ushort_as_half(static_cast<unsigned short>(bits)));
}

// The 16 INT8 values of one 16-byte load, in memory order.
__device__ inline void unpack_int8(uint4 load, float* out) {
  const unsigned words[4] = {load.x, load.y, load.z, load.w};
  for (int w = 0; w < 4; ++w) {
    for (int b = 0; b < 4; ++b) {
      // Shifting the byte to the top and back down as a signed int extends its sign.
      out[4 * w + b] = static_cast<float>(static_cast<int>(words[w] << (24 - 8 * b)) >> 24);
    }
  }
}

// 16-byte asynchronous copy from global to shared memory (sm_80 and later). With `valid` false
// the source size is 0: the 16 bytes are filled with zeros and global memory is not read.
__device__ inline void cp_async16(void* smem, const void* global, bool valid) {
  const unsigned dst = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(global),
               "r"(valid ? 16 : 0)
               : "memory");
}

__device__ inline void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }

// Waits until at most N committed groups are still in flight.
template <int N>
__device__ inline void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}

inline const __half* as_half(const f16* p) { return reinterpret_cast<const __half*>(p); }
inline __half* as_half(f16* p) { return reinterpret_cast<__half*>(p); }

}  // namespace kvb::kernels
