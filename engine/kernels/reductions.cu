// Kernels with one block per row that reduce across the row: RMSNorm and argmax.

#include <math_constants.h>

#include <climits>

#include "common.cuh"

namespace kvb::kernels {
namespace {

// Each thread squares a strided share of the row, the block sums the shares with warp shuffles,
// and every thread then scales its share by the same factor.
__global__ void rms_norm_kernel(const __half* x, const __half* weight, __half* out, int dim,
                                float eps) {
  __shared__ float scratch[kBlockWarps];
  const std::size_t row = static_cast<std::size_t>(blockIdx.x) * dim;
  float sum_sq = 0.0f;
  for (int d = threadIdx.x; d < dim; d += kBlockThreads) {
    const float value = __half2float(x[row + d]);
    sum_sq += value * value;
  }
  const float scale = rsqrtf(block_sum(sum_sq, scratch) / static_cast<float>(dim) + eps);
  for (int d = threadIdx.x; d < dim; d += kBlockThreads) {
    out[row + d] = __float2half(__half2float(x[row + d]) * scale * __half2float(weight[d]));
  }
}

struct Best {
  float value;
  int index;
};

__device__ inline Best better(Best a, Best b) {
  return (b.value > a.value || (b.value == a.value && b.index < a.index)) ? b : a;
}

__global__ void argmax_kernel(const float* logits, std::int32_t* out, int n_cols) {
  __shared__ Best warp_best[kBlockWarps];
  const float* row = logits + static_cast<std::size_t>(blockIdx.x) * n_cols;
  const int tid = threadIdx.x;
  // Threads past the end of a short row hold a candidate that loses to any real column.
  Best best{-CUDART_INF_F, INT_MAX};
  if (tid < n_cols) best = {row[tid], tid};
  for (int c = tid + kBlockThreads; c < n_cols; c += kBlockThreads) {
    if (row[c] > best.value) best = {row[c], c};
  }
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    const Best other{__shfl_xor_sync(0xffffffffu, best.value, offset),
                     __shfl_xor_sync(0xffffffffu, best.index, offset)};
    best = better(best, other);
  }
  if ((tid & (kWarpSize - 1)) == 0) warp_best[tid / kWarpSize] = best;
  __syncthreads();
  if (tid == 0) {
    for (int warp = 1; warp < kBlockWarps; ++warp) best = better(best, warp_best[warp]);
    out[blockIdx.x] = best.index;
  }
}

}  // namespace

void rms_norm(const f16* x, const f16* weight, f16* out, int n_rows, int dim, float eps,
              cudaStream_t stream) {
  rms_norm_kernel<<<n_rows, kBlockThreads, 0, stream>>>(as_half(x), as_half(weight), as_half(out),
                                                        dim, eps);
  KVB_CUDA(cudaGetLastError());
}

void argmax(const float* logits, std::int32_t* out, int n_rows, int n_cols, cudaStream_t stream) {
  argmax_kernel<<<n_rows, kBlockThreads, 0, stream>>>(logits, out, n_cols);
  KVB_CUDA(cudaGetLastError());
}

}  // namespace kvb::kernels
