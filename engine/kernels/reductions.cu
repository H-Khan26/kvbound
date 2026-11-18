// Kernels with one block per row that reduce across the row: RMSNorm, LayerNorm and argmax.

#include <math_constants.h>

#include <climits>

#include "common.cuh"

namespace kvb::kernels {
namespace {

// Each thread squares a strided share of the row, the block sums the shares with warp shuffles,
// and every thread then scales its share by the same factor. With kAdd, the thread first adds y
// to its share of x and stores the FP16 sum at `sum`, which is x itself, and normalizes the sum;
// it rereads only elements it wrote itself.
template <bool kAdd>
__global__ void rms_norm_kernel(const __half* x, const __half* y, __half* sum, const __half* weight,
                                __half* out, int dim, float eps) {
  __shared__ float scratch[kBlockWarps];
  const std::size_t row = static_cast<std::size_t>(blockIdx.x) * dim;
  float sum_sq = 0.0f;
  for (int d = threadIdx.x; d < dim; d += kBlockThreads) {
    float value = __half2float(x[row + d]);
    if constexpr (kAdd) {
      const __half added = __float2half(value + __half2float(y[row + d]));
      sum[row + d] = added;
      value = __half2float(added);
    }
    sum_sq += value * value;
  }
  const float scale = rsqrtf(block_sum(sum_sq, scratch) / static_cast<float>(dim) + eps);
  for (int d = threadIdx.x; d < dim; d += kBlockThreads) {
    out[row + d] = __float2half(__half2float(x[row + d]) * scale * __half2float(weight[d]));
  }
}

// 8-element chunks of a row one thread of rms_norm_wide_kernel holds.
constexpr int kNormChunks = 4;

// The same norm for rows that split into 16-byte chunks, kNormChunks or fewer per thread: each
// thread loads its chunks once, keeps them in registers for the second pass, and reads and
// writes 8 elements per memory access. A decode step normalizes one row per block, so the
// kernel's time is its chain of dependent memory accesses, and this keeps that chain short.
template <bool kAdd>
__global__ void rms_norm_wide_kernel(const uint4* x, const uint4* y, uint4* sum,
                                     const uint4* weight, uint4* out, int dim, float eps) {
  __shared__ float scratch[kBlockWarps];
  const int n_chunks = dim / 8;
  const std::size_t row = static_cast<std::size_t>(blockIdx.x) * n_chunks;
  float v[kNormChunks][8];
  float sum_sq = 0.0f;
  for (int c = 0; c < kNormChunks; ++c) {
    const int i = threadIdx.x + c * kBlockThreads;
    if (i >= n_chunks) break;
    const uint4 in = x[row + i];
    const __half2* h = reinterpret_cast<const __half2*>(&in);
    uint4 added;
    __half2* a = reinterpret_cast<__half2*>(&added);
    const uint4 extra = kAdd ? y[row + i] : uint4{};
    const __half2* e = reinterpret_cast<const __half2*>(&extra);
    for (int j = 0; j < 4; ++j) {
      float2 f = __half22float2(h[j]);
      if constexpr (kAdd) {
        const float2 g = __half22float2(e[j]);
        a[j] = __floats2half2_rn(f.x + g.x, f.y + g.y);
        f = __half22float2(a[j]);
      }
      v[c][2 * j] = f.x;
      v[c][2 * j + 1] = f.y;
      sum_sq += f.x * f.x + f.y * f.y;
    }
    if constexpr (kAdd) sum[row + i] = added;
  }
  const float scale = rsqrtf(block_sum(sum_sq, scratch) / static_cast<float>(dim) + eps);
  for (int c = 0; c < kNormChunks; ++c) {
    const int i = threadIdx.x + c * kBlockThreads;
    if (i >= n_chunks) break;
    const uint4 wl = weight[i];
    const __half2* w = reinterpret_cast<const __half2*>(&wl);
    uint4 result;
    __half2* r = reinterpret_cast<__half2*>(&result);
    for (int j = 0; j < 4; ++j) {
      const float2 wf = __half22float2(w[j]);
      r[j] = __floats2half2_rn(v[c][2 * j] * scale * wf.x, v[c][2 * j + 1] * scale * wf.y);
    }
    out[row + i] = result;
  }
}

bool fits_wide_norm(int dim) { return dim % 8 == 0 && dim / 8 <= kNormChunks * kBlockThreads; }

template <bool kAdd>
void launch_norm(const f16* x, const f16* y, f16* sum, const f16* weight, f16* out, int n_rows,
                 int dim, float eps, cudaStream_t stream) {
  if (fits_wide_norm(dim)) {
    rms_norm_wide_kernel<kAdd><<<n_rows, kBlockThreads, 0, stream>>>(
        reinterpret_cast<const uint4*>(x), reinterpret_cast<const uint4*>(y),
        reinterpret_cast<uint4*>(sum), reinterpret_cast<const uint4*>(weight),
        reinterpret_cast<uint4*>(out), dim, eps);
  } else {
    rms_norm_kernel<kAdd><<<n_rows, kBlockThreads, 0, stream>>>(
        as_half(x), as_half(y), as_half(sum), as_half(weight), as_half(out), dim, eps);
  }
  KVB_CUDA(cudaGetLastError());
}

// LayerNorm in the shape of rms_norm_kernel, with one more pass: the block sums the row for its
// mean, then the squared distances from the mean for its variance, and every thread writes its
// share shifted, scaled, and with the bias added. Taking the variance about the mean, not as
// mean(x^2) - mean^2, keeps its digits when the mean is large against the spread.
template <bool kAdd>
__global__ void layer_norm_kernel(const __half* x, const __half* y, __half* sum,
                                  const __half* weight, const __half* bias, __half* out, int dim,
                                  float eps) {
  __shared__ float scratch[kBlockWarps];
  const std::size_t row = static_cast<std::size_t>(blockIdx.x) * dim;
  float total = 0.0f;
  for (int d = threadIdx.x; d < dim; d += kBlockThreads) {
    float value = __half2float(x[row + d]);
    if constexpr (kAdd) {
      const __half added = __float2half(value + __half2float(y[row + d]));
      sum[row + d] = added;
      value = __half2float(added);
    }
    total += value;
  }
  const float mean = block_sum(total, scratch) / static_cast<float>(dim);
  float sum_sq = 0.0f;
  for (int d = threadIdx.x; d < dim; d += kBlockThreads) {
    const float centered = __half2float(x[row + d]) - mean;
    sum_sq += centered * centered;
  }
  const float scale = rsqrtf(block_sum(sum_sq, scratch) / static_cast<float>(dim) + eps);
  for (int d = threadIdx.x; d < dim; d += kBlockThreads) {
    const float centered = __half2float(x[row + d]) - mean;
    out[row + d] = __float2half(centered * scale * __half2float(weight[d]) + __half2float(bias[d]));
  }
}

// LayerNorm in the shape of rms_norm_wide_kernel: a thread's chunks stay in registers through
// all three passes.
template <bool kAdd>
__global__ void layer_norm_wide_kernel(const uint4* x, const uint4* y, uint4* sum,
                                       const uint4* weight, const uint4* bias, uint4* out, int dim,
                                       float eps) {
  __shared__ float scratch[kBlockWarps];
  const int n_chunks = dim / 8;
  const std::size_t row = static_cast<std::size_t>(blockIdx.x) * n_chunks;
  float v[kNormChunks][8];
  float total = 0.0f;
  for (int c = 0; c < kNormChunks; ++c) {
    const int i = threadIdx.x + c * kBlockThreads;
    if (i >= n_chunks) break;
    const uint4 in = x[row + i];
    const __half2* h = reinterpret_cast<const __half2*>(&in);
    uint4 added;
    __half2* a = reinterpret_cast<__half2*>(&added);
    const uint4 extra = kAdd ? y[row + i] : uint4{};
    const __half2* e = reinterpret_cast<const __half2*>(&extra);
    for (int j = 0; j < 4; ++j) {
      float2 f = __half22float2(h[j]);
      if constexpr (kAdd) {
        const float2 g = __half22float2(e[j]);
        a[j] = __floats2half2_rn(f.x + g.x, f.y + g.y);
        f = __half22float2(a[j]);
      }
      v[c][2 * j] = f.x;
      v[c][2 * j + 1] = f.y;
      total += f.x + f.y;
    }
    if constexpr (kAdd) sum[row + i] = added;
  }
  const float mean = block_sum(total, scratch) / static_cast<float>(dim);
  float sum_sq = 0.0f;
  for (int c = 0; c < kNormChunks; ++c) {
    if (threadIdx.x + c * kBlockThreads >= n_chunks) break;
    for (int j = 0; j < 8; ++j) {
      v[c][j] -= mean;
      sum_sq += v[c][j] * v[c][j];
    }
  }
  const float scale = rsqrtf(block_sum(sum_sq, scratch) / static_cast<float>(dim) + eps);
  for (int c = 0; c < kNormChunks; ++c) {
    const int i = threadIdx.x + c * kBlockThreads;
    if (i >= n_chunks) break;
    const uint4 wl = weight[i], bl = bias[i];
    const __half2* w = reinterpret_cast<const __half2*>(&wl);
    const __half2* b = reinterpret_cast<const __half2*>(&bl);
    uint4 result;
    __half2* r = reinterpret_cast<__half2*>(&result);
    for (int j = 0; j < 4; ++j) {
      const float2 wf = __half22float2(w[j]), bf = __half22float2(b[j]);
      r[j] = __floats2half2_rn(v[c][2 * j] * scale * wf.x + bf.x,
                               v[c][2 * j + 1] * scale * wf.y + bf.y);
    }
    out[row + i] = result;
  }
}

template <bool kAdd>
void launch_layer_norm(const f16* x, const f16* y, f16* sum, const f16* weight, const f16* bias,
                       f16* out, int n_rows, int dim, float eps, cudaStream_t stream) {
  if (fits_wide_norm(dim)) {
    layer_norm_wide_kernel<kAdd><<<n_rows, kBlockThreads, 0, stream>>>(
        reinterpret_cast<const uint4*>(x), reinterpret_cast<const uint4*>(y),
        reinterpret_cast<uint4*>(sum), reinterpret_cast<const uint4*>(weight),
        reinterpret_cast<const uint4*>(bias), reinterpret_cast<uint4*>(out), dim, eps);
  } else {
    layer_norm_kernel<kAdd><<<n_rows, kBlockThreads, 0, stream>>>(
        as_half(x), as_half(y), as_half(sum), as_half(weight), as_half(bias), as_half(out), dim,
        eps);
  }
  KVB_CUDA(cudaGetLastError());
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
  launch_norm<false>(x, nullptr, nullptr, weight, out, n_rows, dim, eps, stream);
}

void add_rms_norm(f16* x, const f16* y, const f16* weight, f16* out, int n_rows, int dim, float eps,
                  cudaStream_t stream) {
  launch_norm<true>(x, y, x, weight, out, n_rows, dim, eps, stream);
}

void layer_norm(const f16* x, const f16* weight, const f16* bias, f16* out, int n_rows, int dim,
                float eps, cudaStream_t stream) {
  launch_layer_norm<false>(x, nullptr, nullptr, weight, bias, out, n_rows, dim, eps, stream);
}

void add_layer_norm(f16* x, const f16* y, const f16* weight, const f16* bias, f16* out, int n_rows,
                    int dim, float eps, cudaStream_t stream) {
  launch_layer_norm<true>(x, y, x, weight, bias, out, n_rows, dim, eps, stream);
}

void argmax(const float* logits, std::int32_t* out, int n_rows, int n_cols, cudaStream_t stream) {
  argmax_kernel<<<n_rows, kBlockThreads, 0, stream>>>(logits, out, n_cols);
  KVB_CUDA(cudaGetLastError());
}

}  // namespace kvb::kernels
